// Terminal sensitivity graph for Gaussian Volterra paths using shared FFTs.
#pragma once

#include "common/check_cuda.cuh"
#include "common/monte_carlo/price_gradients/node_graph/row_preparation.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/reconstruction.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/result_index.cuh"
#include "common/volterra/hybrid_fft_pricer.cuh"
#include "common/volterra/price_gradients/terminal_node_policy.cuh"
#include "common/volterra/price_gradients/gaussian_path_node_evaluation.cuh"
#include "common/workspace_layout.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_factory::workbench::volterra::price_gradients {

template<typename KernelPolicy>
struct GaussianFftKernelEntry {
    using Parameters = typename KernelPolicy::Parameters;
    Parameters parameters{};
    float time_step = 0.0f;
    std::uint32_t maximum_steps = 0U;
    std::uint32_t fft_length = 0U;
};

template<typename KernelPolicy>
class GaussianFftKernelCache {
public:
    using Parameters = typename KernelPolicy::Parameters;
    using Entry = GaussianFftKernelEntry<KernelPolicy>;
    static_assert(std::is_trivially_copyable_v<Parameters>);

    std::uint32_t index(
        const Parameters& parameters,
        float time_step,
        std::uint32_t step_count
    ) {
        if (!std::isfinite(time_step) || !(time_step > 0.0f)
            || step_count == 0U
            || step_count > kMaximumHybridFftStepCount) {
            throw std::invalid_argument("Invalid Gaussian-FFT graph grid.");
        }
        std::uint32_t length = 0U;
        tuning::dispatch_hybrid_fft_specialization<kMaximumHybridFftLength>(
            step_count, [&]<typename Specialization>() {
                length = Specialization::kLength;
            }
        );
        // The low step_count convolution coefficients do not depend on
        // weights beyond that point. Fill the spectrum up to Length/2 and
        // reuse it for every maturity in this FFT-length bucket.
        std::string key(
            reinterpret_cast<const char*>(&parameters), sizeof(Parameters)
        );
        const auto grid_bits = std::bit_cast<std::uint32_t>(time_step);
        key.append(reinterpret_cast<const char*>(&grid_bits),
                   sizeof(grid_bits));
        key.append(reinterpret_cast<const char*>(&length), sizeof(length));
        if (const auto found = indices_.find(key);
            found != indices_.end()) return found->second;
        if (entries_.size() >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Gaussian FFT cache index overflow.");
        }
        const auto result = static_cast<std::uint32_t>(entries_.size());
        entries_.push_back({parameters, time_step, length / 2U, length});
        indices_.emplace(std::move(key), result);
        return result;
    }

    const std::vector<Entry>& entries() const noexcept { return entries_; }

private:
    std::unordered_map<std::string, std::uint32_t> indices_;
    std::vector<Entry> entries_;
};

template<typename KernelPolicy>
struct GaussianFftDeviceKernel {
    typename KernelPolicy::PreparedKernel kernel{};
    float2* spectrum = nullptr;
    float* variances = nullptr;
};

template<typename Scenario, std::size_t MaximumSensitivities>
struct GaussianFftGraphRow {
    static constexpr std::size_t kNodeCapacity =
        mcpg::terminal_node_graph_node_capacity<MaximumSensitivities>();
    Scenario scenarios[kNodeCapacity]{};
    std::uint32_t kernel_indices[kNodeCapacity]{};
    pg::SensitivityStencil<4U> stencils[MaximumSensitivities]{};
    mcpg::SensitivityNodeIndices<4U>
        node_indices[MaximumSensitivities]{};
    std::uint16_t node_count = 0U;
    std::uint32_t maximum_steps = 0U;
};

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename HostPlan,
    typename KernelCache>
auto prepare_gaussian_fft_graph_row(
    const HostPlan& host,
    std::size_t row,
    KernelCache& cache
) {
    using Preparation = typename HostPlan::Preparation;
    using Adapter = typename Preparation::ModelAdapter;
    using Row = GaussianFftGraphRow<
        typename Preparation::Scenario, MaximumSensitivities
    >;
    if (row >= host.result_count
        || host.sensitivity_count() == 0U
        || host.sensitivity_count() > MaximumSensitivities) {
        throw std::invalid_argument("Invalid Gaussian-FFT graph row.");
    }
    const auto indices = decode_model_product_result_index(
        row, host.products.size(), host.construction
    );
    typename Preparation::Scenario central{};
    if (!Preparation::make_central(
            host.models[indices.model_index],
            host.products[indices.product_index],
            host.time, central
        )) {
        throw std::invalid_argument("Invalid Gaussian-FFT central row.");
    }
    Row result{};
    int error = pg::device_preparation::valid;
    std::size_t error_sensitivity = 0U;
    if (!mcpg::node_graph_detail::prepare_sensitivity_row_from_central<
            Orders, Preparation, MaximumSensitivities
        >(
            central, host.sensitivities.data(), host.sensitivity_count(),
            host.time, result.scenarios, result.stencils,
            result.node_indices, result.node_count, result.maximum_steps,
            error, error_sensitivity
        )) {
        throw std::invalid_argument(
            "Invalid Gaussian-FFT sensitivity node "
            + std::to_string(error_sensitivity)
            + " (error " + std::to_string(error) + ")."
        );
    }
    for (std::size_t node = 0U; node < result.node_count; ++node) {
        const auto& scenario = result.scenarios[node];
        result.kernel_indices[node] = cache.index(
            Adapter::kernel_parameters(scenario.model),
            host.time.dt, scenario.step_count
        );
    }
    return result;
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename HostPlan,
    typename KernelCache>
void prepare_gaussian_fft_graph_cache(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch,
    KernelCache& cache
) {
    if (launch.result_offset > host.result_count
        || launch.result_count > host.result_count - launch.result_offset) {
        throw std::invalid_argument("Gaussian-FFT graph rows exceed host plan.");
    }
    // For a full Cartesian request the model bump endpoints and the product
    // grids factor independently. Build O(models + products) rows on the
    // host, then combine only their distinct kernel/grid keys. This avoids
    // rebuilding a million stencils just to size one shared FFT cache.
    if (host.construction == PriceConstruction::CartesianProduct
        && launch.result_offset == 0U
        && launch.result_count == host.result_count) {
        using Adapter = typename HostPlan::Preparation::ModelAdapter;
        using Parameters = typename KernelCache::Parameters;
        std::unordered_set<std::uint32_t> central_steps;
        std::unordered_set<std::uint32_t> maturity_steps;
        for (std::size_t product = 0U;
             product < host.products.size(); ++product) {
            const auto row = prepare_gaussian_fft_graph_row<
                Orders, MaximumSensitivities
            >(host, product, cache);
            central_steps.insert(row.scenarios[0U].step_count);
            for (std::size_t node = 1U; node < row.node_count; ++node) {
                if (row.scenarios[node].step_count
                    != row.scenarios[0U].step_count) {
                    maturity_steps.insert(
                        row.scenarios[node].step_count
                    );
                }
            }
        }
        for (std::size_t model = 0U;
             model < host.models.size(); ++model) {
            const auto row = prepare_gaussian_fft_graph_row<
                Orders, MaximumSensitivities
            >(host, model * host.products.size(), cache);
            std::vector<Parameters> kernel_parameters;
            kernel_parameters.reserve(row.node_count);
            for (std::size_t node = 0U; node < row.node_count; ++node) {
                kernel_parameters.push_back(Adapter::kernel_parameters(
                    row.scenarios[node].model
                ));
            }
            for (const auto steps : central_steps)
                for (const auto& parameters : kernel_parameters)
                    (void)cache.index(parameters, host.time.dt, steps);
            const auto central_parameters =
                Adapter::kernel_parameters(host.models[model]);
            for (const auto steps : maturity_steps)
                (void)cache.index(
                    central_parameters, host.time.dt, steps
                );
        }
        return;
    }
    for (std::size_t row = launch.result_offset;
         row < launch.result_offset + launch.result_count; ++row) {
        (void)prepare_gaussian_fft_graph_row<
            Orders, MaximumSensitivities
        >(host, row, cache);
    }
}

struct GaussianTerminalObserver {
    template<typename State>
    __device__ __forceinline__ bool on_observation(
        std::uint32_t, const State&
    ) {
        return true;
    }
};

template<typename KernelPolicy, typename ModelPathPolicy,
         typename ProductPolicy>
struct GaussianTerminalEvaluation {
    using NodePolicy = PreparedLiftTerminalNodePolicy<
        ModelPathPolicy, ProductPolicy
    >;
    static constexpr bool kReuseCentralValue = true;
    static constexpr bool kBatchSharedPath = false;

    template<typename TerminalRow, typename Scenario>
    __device__ static float evaluate(
        const TerminalRow& row,
        const typename ModelPathPolicy::PreparedModel& model,
        const Scenario&,
        const typename NodePolicy::Metadata&,
        pg::TimeConfiguration,
        std::size_t path,
        const float* variances,
        hybrid_fft::FftPathConvolution convolution
    ) {
        GaussianTerminalObserver observer{};
        const auto state = hybrid_fft::simulate_observed_path<
            ModelPathPolicy
        >(row, model, path, variances, convolution, observer);
        return ModelPathPolicy::spot(state);
    }
};

template<typename NodePolicy>
struct GaussianFftGraphExecutionPlan {
    mcpg::TerminalNodeGraphConfiguration graph{};
    mcpg::TerminalNodeGraphWorkspaceLayout node_layout{};
    std::size_t fft_offset = 0U;
    std::size_t row_offset = 0U;
    std::size_t kernel_offset = 0U;
    std::size_t spectrum_offset = 0U;
    std::size_t variance_offset = 0U;
    std::size_t kernel_capacity = 0U;
    std::size_t spectrum_capacity = 0U;
    std::size_t variance_capacity = 0U;
    std::size_t maximum_steps = 0U;
    std::size_t bytes = 0U;
};

template<
    typename KernelPolicy,
    typename ModelPathPolicy,
    typename ProductPolicy,
    typename HostPlan,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename Evaluation = GaussianTerminalEvaluation<
        KernelPolicy, ModelPathPolicy, ProductPolicy>>
auto plan_gaussian_fft_node_graph(
    const HostPlan& host,
    GaussianFftKernelCache<KernelPolicy>& cache,
    const pg::LaunchConfiguration& launch,
    std::size_t workspace_limit = 1U << 29U
) {
    static_assert(Orders != pg::SensitivityOrders::none);
    using NodePolicy = typename Evaluation::NodePolicy;
    using Row = GaussianFftGraphRow<
        typename HostPlan::Preparation::Scenario, MaximumSensitivities
    >;
    using DeviceKernel = GaussianFftDeviceKernel<KernelPolicy>;
    if (host.request.orders != Orders
        || launch.result_count == 0U
        || launch.paths_per_price < 2U) {
        throw std::invalid_argument("Invalid Gaussian-FFT graph request.");
    }
    prepare_gaussian_fft_graph_cache<Orders, MaximumSensitivities>(
        host, launch, cache
    );
    if (cache.entries().empty()) {
        throw std::invalid_argument("Empty Gaussian-FFT kernel cache.");
    }
    std::size_t maximum_steps = 0U;
    std::size_t spectrum_count = 0U;
    std::size_t variance_count = 0U;
    for (const auto& entry : cache.entries()) {
        maximum_steps = std::max(maximum_steps,
                                 std::size_t(entry.maximum_steps));
        spectrum_count = checked_hybrid_fft_sum(
            spectrum_count, entry.fft_length,
            "Gaussian-FFT spectrum cache overflows size_t."
        );
        variance_count = checked_hybrid_fft_sum(
            variance_count, entry.maximum_steps,
            "Gaussian-FFT variance cache overflows size_t."
        );
    }
    // One row at a time keeps the expensive convolution buffer bounded.
    constexpr std::size_t fft_threads = tuning::kPricingPathThreads;
    std::size_t path_chunk = std::min<std::size_t>(
        2048U,
        std::max<std::size_t>(
            fft_threads,
            launch.paths_per_price / fft_threads * fft_threads
        )
    );
    while (true) {
        mcpg::TerminalNodeGraphConfiguration graph{1U, path_chunk, 1U};
        const auto layout = mcpg::terminal_node_graph_workspace_layout<
            NodePolicy, Orders, MaximumSensitivities
        >(host.sensitivity_count(), launch.threads_per_block, graph);
        std::size_t bytes = layout.bytes;
        const auto fft_offset =
            workspace_layout::append_array<unsigned char>(
                bytes,
                required_hybrid_fft_workspace_bytes(
                    maximum_steps, launch.paths_per_price, path_chunk
                ),
                "Gaussian-FFT graph workspace overflow."
            );
        const auto row_offset = workspace_layout::append_array<Row>(
            bytes, 1U, "Gaussian-FFT graph workspace overflow."
        );
        const auto kernel_offset =
            workspace_layout::append_array<DeviceKernel>(
                bytes, cache.entries().size(),
                "Gaussian-FFT kernel cache workspace overflow."
            );
        const auto spectrum_offset =
            workspace_layout::append_array<float2>(
                bytes, spectrum_count,
                "Gaussian-FFT spectrum cache workspace overflow."
            );
        const auto variance_offset =
            workspace_layout::append_array<float>(
                bytes, variance_count,
                "Gaussian-FFT variance cache workspace overflow."
            );
        if (bytes <= workspace_limit) {
            return GaussianFftGraphExecutionPlan<NodePolicy>{
                graph, layout, fft_offset, row_offset, kernel_offset,
                spectrum_offset, variance_offset, cache.entries().size(),
                spectrum_count, variance_count, maximum_steps, bytes
            };
        }
        if (path_chunk <= fft_threads) {
            throw std::invalid_argument(
                "Gaussian-FFT graph exceeds its workspace limit."
            );
        }
        path_chunk = std::max<std::size_t>(
            fft_threads,
            path_chunk / (2U * fft_threads) * fft_threads
        );
    }
}

template<typename KernelPolicy, unsigned int Length, typename KernelForward>
__global__ void prepare_gaussian_fft_kernel_cache_entry(
    typename KernelPolicy::Parameters parameters,
    float dt,
    std::uint32_t maximum_steps,
    GaussianFftDeviceKernel<KernelPolicy>* output
) {
    using Complex = typename KernelForward::value_type;
    Complex data[KernelForward::storage_size];
    __shared__ typename KernelPolicy::PreparedKernel shared_kernel;
    if (threadIdx.x == 0U && threadIdx.y == 0U) {
        shared_kernel = KernelPolicy::prepare(parameters, dt);
        output->kernel = shared_kernel;
    }
    __syncthreads();
    const auto kernel = shared_kernel;
    if (KernelForward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U;
             item < KernelForward::input_ept; ++item) {
            const unsigned int index =
                item * KernelForward::stride + threadIdx.x;
            const float weight = index + 1U < maximum_steps
                ? KernelPolicy::far_cell_weight(kernel, index + 2U)
                : 0.0f;
            reinterpret_cast<float2*>(data)[item] = {weight, 0.0f};
            if (index < maximum_steps) {
                output->variances[index] =
                    KernelPolicy::volterra_variance(
                        kernel, static_cast<float>(index + 1U) * dt
                    );
            }
        }
    }
    extern __shared__ __align__(16) unsigned char shared_storage[];
    KernelForward().execute(
        data, reinterpret_cast<Complex*>(shared_storage)
    );
    if (KernelForward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U;
             item < KernelForward::output_ept; ++item) {
            const unsigned int index =
                item * KernelForward::stride + threadIdx.x;
            if (index < Length)
                output->spectrum[index] =
                    reinterpret_cast<float2*>(data)[item];
        }
    }
}

template<typename KernelPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename Scenario>
__global__ void prepare_gaussian_fft_group_row_kernel(
    const Scenario* scenario,
    const GaussianFftDeviceKernel<KernelPolicy>* kernel_cache,
    std::size_t global_row,
    std::uint64_t base_seed,
    float dt,
    hybrid_fft::PreparedRow<
        KernelPolicy, ModelPathPolicy, ProductPolicy,
        TerminalHybridSchedule
    >* prepared_row
) {
    if (threadIdx.x != 0U) return;
    const auto s = *scenario;
    using Row = hybrid_fft::PreparedRow<
        KernelPolicy, ModelPathPolicy, ProductPolicy,
        TerminalHybridSchedule
    >;
    *prepared_row = Row{
        kernel_cache->kernel,
        {},
        {},
        {s.maturity_years, dt, s.step_count},
        philox::make_key(base_seed + global_row),
        sqrtf(dt)
    };
}

template<typename NodePolicy, typename Scenario,
         std::size_t MaximumSensitivities>
__global__ void initialize_gaussian_fft_graph_row_kernel(
    const GaussianFftGraphRow<Scenario, MaximumSensitivities>* row,
    std::size_t global_row,
    std::size_t sensitivity_count,
    pg::TimeConfiguration time,
    mcpg::TerminalNodeGraphWorkspace<NodePolicy> workspace,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs
) {
    if (threadIdx.x != 0U) return;
    workspace.row_status[0] = 1U;
    for (std::size_t node = 0U; node < row->node_count; ++node) {
        workspace.node_metadata[node] =
            NodePolicy::prepare_metadata(row->scenarios[node], time);
    }
    for (std::size_t sensitivity = 0U;
         sensitivity < sensitivity_count; ++sensitivity) {
        workspace.node_indices[sensitivity] =
            row->node_indices[sensitivity];
        stencil_outputs.stencils[
            global_row * sensitivity_count + sensitivity
        ] = row->stencils[sensitivity];
    }
}


template<typename KernelPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename Scenario,
         std::size_t MaximumSensitivities,
         typename Evaluation = GaussianTerminalEvaluation<
             KernelPolicy, ModelPathPolicy, ProductPolicy>>
__global__ void evaluate_gaussian_fft_group_nodes_kernel(
    const GaussianFftGraphRow<Scenario, MaximumSensitivities>* row,
    std::uint32_t kernel_index,
    std::uint32_t step_count,
    pg::TimeConfiguration time,
    const hybrid_fft::PreparedRow<
        KernelPolicy, ModelPathPolicy, ProductPolicy,
        TerminalHybridSchedule
    >* prepared_row,
    const float* variances,
    const float2* convolutions,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    mcpg::TerminalNodeGraphWorkspace<
        typename Evaluation::NodePolicy
    > workspace
) {
    constexpr std::size_t capacity =
        GaussianFftGraphRow<Scenario, MaximumSensitivities>::kNodeCapacity;
    using PreparedModel = typename ModelPathPolicy::PreparedModel;
    using NodePolicy = typename Evaluation::NodePolicy;
    __shared__ PreparedModel prepared_models[capacity];
    __shared__ typename NodePolicy::Metadata shared_metadata[capacity];
    __shared__ hybrid_fft::PreparedRow<
        KernelPolicy, ModelPathPolicy, ProductPolicy, TerminalHybridSchedule
    > shared_row;
    if (threadIdx.x == 0U) shared_row = *prepared_row;
    __syncthreads();
    for (std::size_t node = threadIdx.x; node < row->node_count;
         node += blockDim.x) {
        shared_metadata[node] = workspace.node_metadata[node];
        if (row->kernel_indices[node] == kernel_index
            && row->scenarios[node].step_count == step_count
            && (node == 0U || !row->scenarios[node].reuse_central
                || !Evaluation::kReuseCentralValue)) {
            auto model = row->scenarios[node].model;
            model.spot = row->scenarios[node].simulation_spot;
            prepared_models[node] = ModelPathPolicy::prepare_model(
                model, shared_row.schedule.time_step
            );
        }
    }
    __syncthreads();
    const auto local_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_path >= path_count) return;
    const auto path = first_path + local_path;
    const auto convolution = hybrid_fft::FftPathConvolution{
        convolutions, local_path / 2U, (local_path & 1U) != 0U
    };
    auto* values = workspace.node_values + local_path * capacity;
    if constexpr (Evaluation::kBatchSharedPath) {
        if (kernel_index == row->kernel_indices[0U]
            && step_count == row->scenarios[0U].step_count) {
            constexpr std::size_t batch_capacity = 8U;
            std::uint16_t indices[batch_capacity]{};
            float batch_values[batch_capacity]{};
            std::size_t scan = 0U;
            while (scan < row->node_count) {
                std::size_t count = 0U;
                for (; scan < row->node_count && count < batch_capacity;
                     ++scan) {
                    if (scan == 0U || row->scenarios[scan].reuse_central)
                        indices[count++] = static_cast<std::uint16_t>(scan);
                }
                if (count == 0U) continue;
                Evaluation::evaluate_batch(
                    shared_row, prepared_models[0U], row->scenarios,
                    shared_metadata, indices, count, time, path, variances,
                    convolution, batch_values
                );
                for (std::size_t i = 0U; i < count; ++i)
                    values[indices[i]] = batch_values[i];
            }
        }
        for (std::size_t node = 1U; node < row->node_count; ++node) {
            if (row->kernel_indices[node] != kernel_index
                || row->scenarios[node].step_count != step_count
                || row->scenarios[node].reuse_central) continue;
            values[node] = Evaluation::evaluate(
                shared_row, prepared_models[node], row->scenarios[node],
                shared_metadata[node], time, path, variances, convolution
            );
        }
    } else {
        for (std::size_t node = 0U; node < row->node_count; ++node) {
            if (row->kernel_indices[node] != kernel_index
                || row->scenarios[node].step_count != step_count) continue;
            float spot = 0.0f;
            if (node != 0U && row->scenarios[node].reuse_central) {
                spot = values[0U];
            } else {
                spot = Evaluation::evaluate(
                    shared_row, prepared_models[node], row->scenarios[node],
                    shared_metadata[node], time, path, variances, convolution
                );
            }
            values[node] = spot;
        }
    }
}

template<typename KernelPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename Evaluation,
         unsigned int Length,
         unsigned int ElementsPerThread, unsigned int FftsPerBlock,
         typename Scenario, std::size_t MaximumSensitivities>
void launch_gaussian_fft_convolution_group(
    const GaussianFftGraphRow<Scenario, MaximumSensitivities>* device_row,
    std::size_t representative,
    std::size_t global_row,
    std::uint32_t kernel_index,
    std::uint32_t step_count,
    const GaussianFftDeviceKernel<KernelPolicy>* device_kernel,
    const float2* spectrum,
    const float* variances,
    hybrid_fft::PreparedRow<
        KernelPolicy, ModelPathPolicy, ProductPolicy,
        TerminalHybridSchedule
    >* prepared_row,
    float2* convolutions,
    mcpg::TerminalNodeGraphWorkspace<
        typename Evaluation::NodePolicy
    > node_workspace,
    std::size_t first_path,
    std::size_t count,
    std::size_t path_capacity,
    std::uint64_t seed,
    pg::TimeConfiguration time
) {
    using Types = hybrid_fft::FftTypes<
        Length, ElementsPerThread, FftsPerBlock
    >;
    using Forward = typename Types::Forward;
    using Inverse = typename Types::Inverse;
    constexpr std::size_t shared = std::max(
        std::size_t(Forward::shared_memory_size),
        std::size_t(Inverse::shared_memory_size)
    );
    static const bool configured = [] {
        check_cuda(cudaFuncSetAttribute(
            hybrid_fft::convolve_paths_kernel<
                KernelPolicy, ModelPathPolicy, ProductPolicy,
                TerminalHybridSchedule, Length, Forward, Inverse
            >,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared)
        ), "Gaussian graph FFT convolution shared-memory opt-in");
        return true;
    }();
    (void)configured;
    prepare_gaussian_fft_group_row_kernel<
        KernelPolicy, ModelPathPolicy, ProductPolicy, Scenario
    ><<<1U, 1U>>>(
        &device_row->scenarios[representative],
        device_kernel, global_row, seed, time.dt, prepared_row
    );
    check_cuda(cudaGetLastError(), "Gaussian graph group row preparation");
    const auto pairs = hybrid_fft_ceiling_division(count, 2U);
    const auto blocks = hybrid_fft_ceiling_division(
        pairs, FftsPerBlock
    );
    hybrid_fft::convolve_paths_kernel<
        KernelPolicy, ModelPathPolicy, ProductPolicy,
        TerminalHybridSchedule, Length, Forward, Inverse
    ><<<static_cast<unsigned int>(blocks), Forward::block_dim, shared>>>(
        first_path, count, prepared_row, spectrum, convolutions
    );
    check_cuda(cudaGetLastError(), "Gaussian graph FFT convolution");
    evaluate_gaussian_fft_group_nodes_kernel<
        KernelPolicy, ModelPathPolicy, ProductPolicy,
        Scenario, MaximumSensitivities, Evaluation
    ><<<static_cast<unsigned int>(
        hybrid_fft_ceiling_division(count, tuning::kPricingPathThreads)
    ), tuning::kPricingPathThreads>>>(
        device_row, kernel_index, step_count, time, prepared_row,
        variances, convolutions,
        first_path, count, path_capacity, node_workspace
    );
    check_cuda(cudaGetLastError(), "Gaussian graph path evaluation");
}

template<typename KernelPolicy, typename ModelPathPolicy,
         typename ProductPolicy>
void prepare_gaussian_fft_kernel_cache(
    const GaussianFftKernelCache<KernelPolicy>& cache,
    GaussianFftDeviceKernel<KernelPolicy>* device_cache
) {
    for (std::size_t index = 0U; index < cache.entries().size(); ++index) {
        const auto& entry = cache.entries()[index];
        tuning::dispatch_hybrid_fft_specialization<kMaximumHybridFftLength>(
            entry.maximum_steps, [&]<typename Specialization>() {
                constexpr auto length = Specialization::kLength;
                using Types = hybrid_fft::FftTypes<
                    length,
                    Specialization::kPricingElementsPerThread,
                    1U
                >;
                using Forward = typename Types::Forward;
                constexpr std::size_t shared =
                    Forward::shared_memory_size;
                static const bool configured = [] {
                    check_cuda(cudaFuncSetAttribute(
                        prepare_gaussian_fft_kernel_cache_entry<
                            KernelPolicy, length, Forward
                        >,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(shared)
                    ), "Gaussian graph FFT preparation shared-memory opt-in");
                    return true;
                }();
                (void)configured;
                prepare_gaussian_fft_kernel_cache_entry<
                    KernelPolicy, length, Forward
                ><<<1U, Forward::block_dim, shared>>>(
                    entry.parameters, entry.time_step,
                    entry.maximum_steps, device_cache + index
                );
                check_cuda(cudaGetLastError(),
                           "Gaussian graph FFT cache preparation");
            }
        );
    }
}

template<
    typename KernelPolicy, typename ModelPathPolicy,
    typename ProductPolicy, typename HostPlan,
    pg::SensitivityOrders Orders, std::size_t MaximumSensitivities,
    typename Evaluation = GaussianTerminalEvaluation<
        KernelPolicy, ModelPathPolicy, ProductPolicy>>
void launch_gaussian_fft_node_graph(
    const HostPlan& host,
    GaussianFftKernelCache<KernelPolicy>& cache,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    const GaussianFftGraphExecutionPlan<
        typename Evaluation::NodePolicy
    >& execution,
    void* storage,
    std::size_t storage_bytes
) {
    static_assert(Orders != pg::SensitivityOrders::none);
    using NodePolicy = typename Evaluation::NodePolicy;
    using Scenario = typename HostPlan::Preparation::Scenario;
    using Row = GaussianFftGraphRow<Scenario, MaximumSensitivities>;
    using PreparedRow = hybrid_fft::PreparedRow<
        KernelPolicy, ModelPathPolicy, ProductPolicy,
        TerminalHybridSchedule
    >;
    using DeviceKernel = GaussianFftDeviceKernel<KernelPolicy>;
    if (!storage || storage_bytes < execution.bytes
        || host.request.orders != Orders
        || cache.entries().size() != execution.kernel_capacity
        || launch.result_offset > host.result_count
        || launch.result_count > host.result_count - launch.result_offset
        || launch.paths_per_price < 2U
        || execution.graph.row_chunk_size != 1U
        || stencil_outputs.stencils == nullptr
        || stencil_outputs.capacity <
            host.result_count * host.sensitivity_count()
        || outputs.prices == nullptr
        || outputs.price_standard_errors == nullptr
        || (pg::requests_first_v<Orders>
            && (outputs.gradients == nullptr
                || outputs.gradient_standard_errors == nullptr))
        || (pg::requests_second_v<Orders>
            && (outputs.diagonal_hessians == nullptr
                || outputs.diagonal_hessian_standard_errors == nullptr))
        || outputs.price_capacity < host.result_count
        || outputs.sensitivity_capacity <
            host.result_count * host.sensitivity_count()) {
        throw std::invalid_argument("Invalid Gaussian-FFT graph launch.");
    }
    auto* base = static_cast<unsigned char*>(storage);
    auto* fft_base = base + execution.fft_offset;
    auto* device_row = reinterpret_cast<Row*>(base + execution.row_offset);
    auto* device_cache = reinterpret_cast<DeviceKernel*>(
        base + execution.kernel_offset
    );
    auto* prepared_row = reinterpret_cast<PreparedRow*>(
        fft_base + kHybridFftPreparedRowOffset
    );
    auto* convolutions = reinterpret_cast<float2*>(
        fft_base + kHybridFftConvolutionOffset
    );
    const auto node_workspace =
        mcpg::make_terminal_node_graph_workspace<NodePolicy>(
            storage, storage_bytes, execution.node_layout
        );
    std::vector<DeviceKernel> device_descriptors(cache.entries().size());
    auto* spectra = reinterpret_cast<float2*>(
        base + execution.spectrum_offset
    );
    auto* variances = reinterpret_cast<float*>(
        base + execution.variance_offset
    );
    std::size_t spectrum_cursor = 0U;
    std::size_t variance_cursor = 0U;
    for (std::size_t index = 0U; index < cache.entries().size(); ++index) {
        device_descriptors[index].spectrum = spectra + spectrum_cursor;
        device_descriptors[index].variances = variances + variance_cursor;
        spectrum_cursor += cache.entries()[index].fft_length;
        variance_cursor += cache.entries()[index].maximum_steps;
    }
    if (spectrum_cursor != execution.spectrum_capacity
        || variance_cursor != execution.variance_capacity) {
        throw std::invalid_argument(
            "Gaussian-FFT cache layout changed after planning."
        );
    }
    check_cuda(cudaMemcpy(
        device_cache, device_descriptors.data(),
        device_descriptors.size() * sizeof(DeviceKernel),
        cudaMemcpyHostToDevice
    ), "Gaussian graph cache descriptor upload");
    prepare_gaussian_fft_kernel_cache<
        KernelPolicy, ModelPathPolicy, ProductPolicy
    >(cache, device_cache);
    const auto prepared_count = cache.entries().size();
    const auto output_count =
        mcpg::terminal_node_graph_output_count<Orders>(
            host.sensitivity_count()
        );
    for (std::size_t global_row = launch.result_offset;
         global_row < launch.result_offset + launch.result_count;
         ++global_row) {
        const Row row = prepare_gaussian_fft_graph_row<
            Orders, MaximumSensitivities
        >(host, global_row, cache);
        if (cache.entries().size() != prepared_count) {
            throw std::invalid_argument(
                "Gaussian-FFT cache changed after its spectra were prepared."
            );
        }
        check_cuda(cudaMemcpy(
            device_row, &row, sizeof(Row), cudaMemcpyHostToDevice
        ), "Gaussian graph row upload");
        initialize_gaussian_fft_graph_row_kernel<
            NodePolicy, Scenario, MaximumSensitivities
        ><<<1U, 1U>>>(
            device_row, global_row, host.sensitivity_count(), host.time,
            node_workspace, stencil_outputs
        );
        check_cuda(cudaGetLastError(), "Gaussian graph row initialization");
        std::uint32_t groups[Row::kNodeCapacity]{};
        std::uint32_t group_steps[Row::kNodeCapacity]{};
        std::size_t representatives[Row::kNodeCapacity]{};
        std::size_t group_count = 0U;
        for (std::size_t node = 0U; node < row.node_count; ++node) {
            const auto index = row.kernel_indices[node];
            const auto steps = row.scenarios[node].step_count;
            bool found = false;
            for (std::size_t group = 0U; group < group_count; ++group)
                found = found || (groups[group] == index
                    && group_steps[group] == steps);
            if (!found) {
                groups[group_count] = index;
                group_steps[group_count] = steps;
                representatives[group_count] = node;
                ++group_count;
            }
        }
        for (std::size_t first_path = 0U;
             first_path < launch.paths_per_price;
             first_path += execution.graph.path_chunk_size) {
            const auto count = std::min(
                execution.graph.path_chunk_size,
                launch.paths_per_price - first_path
            );
            for (std::size_t group = 0U; group < group_count; ++group) {
                const auto index = groups[group];
                const auto& entry = cache.entries()[index];
                tuning::dispatch_hybrid_fft_specialization<
                    kMaximumHybridFftLength
                >(entry.maximum_steps, [&]<typename Specialization>() {
                    launch_gaussian_fft_convolution_group<
                        KernelPolicy, ModelPathPolicy, ProductPolicy,
                        Evaluation, Specialization::kLength,
                        Specialization::kPricingElementsPerThread,
                        Specialization::kPricingFftsPerBlock,
                        Scenario, MaximumSensitivities
                    >(
                        device_row, representatives[group], global_row,
                        index, group_steps[group], device_cache + index,
                        device_descriptors[index].spectrum,
                        device_descriptors[index].variances,
                        prepared_row, convolutions, node_workspace,
                        first_path, count,
                        execution.graph.path_chunk_size, launch.base_seed,
                        host.time
                    );
                });
            }
            mcpg::node_graph_detail::accumulate_node_moments_kernel<
                Orders, NodePolicy, MaximumSensitivities
            ><<<dim3(1U, static_cast<unsigned int>(output_count)),
                 launch.threads_per_block>>>(
                global_row, 1U, first_path, count,
                execution.graph.path_chunk_size,
                host.sensitivity_count(), node_workspace, stencil_outputs
            );
            check_cuda(cudaGetLastError(),
                       "Gaussian graph moment accumulation");
        }
        mcpg::node_graph_detail::finalize_node_moments_kernel<
            Orders, NodePolicy
        ><<<dim3(1U, static_cast<unsigned int>(output_count)),
             launch.threads_per_block,
             2U * (launch.threads_per_block / 32U) * sizeof(double)>>>(
            global_row, 1U, host.sensitivity_count(),
            launch.paths_per_price, node_workspace, outputs
        );
        check_cuda(cudaGetLastError(),
                   "Gaussian graph moment finalization");
    }
}

}  // namespace ai_factory::workbench::volterra::price_gradients
