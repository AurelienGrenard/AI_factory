// Shared multi-block VarianceGamma American-option pricing with GPU Longstaff-Schwartz.
#include "model/equity/variance_gamma/american_option.cuh"

#include "common/check_cuda.cuh"
#include "common/result_index.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/longstaff_schwartz/laguerre.cuh"
#include "common/longstaff_schwartz/exercise_schedule.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/longstaff_schwartz/regression.cu"
#include "common/longstaff_schwartz/workspace.cuh"
#include "common/reductions.cuh"

// Include the dynamics implementation so NVCC can inline each time step.
#include "model/equity/variance_gamma/dynamics.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::variance_gamma {
namespace lsm = longstaff_schwartz;

namespace {

using RegressionBasis = lsm::RegressionBasis;

constexpr std::size_t kMomentValueCount = 2U;

template <OptionSide Side>
constexpr const char* product_name() {
    if constexpr (Side == OptionSide::call) return "American-call";
    return "American-put";
}

template <OptionSide Side>
__device__ __forceinline__ float immediate_payoff(float spot, float strike) {
    if constexpr (Side == OptionSide::call) {
        return fmaxf(spot - strike, 0.0f);
    }
    return fmaxf(strike - spot, 0.0f);
}

// Prepared model, schedule, and memory location for one batch price.
struct PreparedRow {
    PreparedModel model;
    PreparedTransition initial_stub_transition;
    PreparedTransition regular_transition;
    philox::PhiloxKey key;
    std::size_t result_index;
    std::size_t state_offset;
    float strike;
    float initial_spot;
    float inverse_strike;
    float exercise_discount;
    float initial_discount;
    std::uint32_t exercise_count;
};

// Name the single model state region returned by the generic layout.
struct StateRegions {
    lsm::WorkspaceRegion spots;
};

lsm::WorkspaceDescriptor workspace_descriptor() {
    return {
        sizeof(PreparedRow),
        alignof(PreparedRow),
        {
            {sizeof(float), alignof(float)},
        },
        RegressionBasis::kSize,
        RegressionBasis::kRegressionValueCount,
    };
}

StateRegions variance_gamma_state_regions(const lsm::WorkspaceLayout& layout) {
    if (layout.state_fields.size() != 1U) {
        throw std::logic_error(
            "Variance-Gamma American options require one state field."
        );
    }
    return {layout.state_fields[0]};
}

std::vector<lsm::EarlyExerciseRowPlan> make_row_plans(
    const product::AmericanOptionParameters* host_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t paths_per_price,
    const char* name
) {
    std::vector<lsm::EarlyExerciseRowPlan> rows;
    rows.reserve(result_count);
    for (std::size_t result_index = 0U;
         result_index < result_count;
         ++result_index) {
        const std::size_t product_index = cartesian_product
            ? result_index % product_count
            : result_index;
        const product::AmericanOptionParameters& product =
            host_products[product_index];
        const std::uint32_t count = lsm::maturity_anchored_exercise_count(
            product.maturity, product.exercise_interval, name
        );
        rows.push_back({
            count,
            checked_workspace_product(
                paths_per_price,
                static_cast<std::size_t>(count - 1U),
                "American-option row state count exceeds size_t."
            ),
        });
    }
    return rows;
}

// Prepare each batch row once before thousands of path blocks consume it.
__global__ void prepare_rows_kernel(
    const ModelParameters* __restrict__ models,
    const product::AmericanOptionParameters* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t batch_size,
    float day_fraction,
    std::uint64_t base_seed,
    std::size_t result_offset,
    const std::uint32_t* __restrict__ exercise_counts,
    const std::size_t* __restrict__ state_offsets,
    PreparedRow* __restrict__ prepared_rows
) {
    const std::size_t batch_price =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch_price >= batch_size) return;

    const std::size_t result_index = result_offset + batch_price;
    const ModelProductIndices indices =
        decode_model_product_result_index(
            result_index, product_count, cartesian_product
        );
    const std::size_t model_index = indices.model_index;
    const std::size_t product_index = indices.product_index;
    const ModelParameters model = models[model_index];
    const product::AmericanOptionParameters product = products[product_index];
    const std::uint32_t row_exercise_count =
        exercise_counts[batch_price];
    const std::uint32_t first_exercise_days = product.maturity
        - (row_exercise_count - 1U) * product.exercise_interval;
    const float first_exercise_time =
        static_cast<float>(first_exercise_days) * day_fraction;
    const float exercise_interval =
        static_cast<float>(product.exercise_interval) * day_fraction;
    const PreparedModel prepared_model = prepare_model(model);
    prepared_rows[batch_price] = {
        prepared_model,
        prepare_transition(prepared_model, first_exercise_time),
        prepare_transition(prepared_model, exercise_interval),
        philox::make_key(base_seed + result_index),
        result_index,
        state_offsets[batch_price],
        product.strike,
        model.spot,
        1.0f / product.strike,
        expf(-model.risk_free_rate * exercise_interval),
        expf(-model.risk_free_rate * first_exercise_time),
        row_exercise_count,
    };
}

// Simulate every path and initialize its undiscounted maturity cashflow.
template <OptionSide Side>
__global__ void simulate_paths_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    float* __restrict__ observed_spots,
    float* __restrict__ cashflows
) {
    __shared__ PreparedRow row;
    if (threadIdx.x == 0U) row = prepared_rows[blockIdx.y];
    __syncthreads();

    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    float* const row_spots = observed_spots + row.state_offset;
    float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;

    // Each warp writes consecutive paths at every grid-stride iteration.
    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const State terminal = simulate_on_regular_grid(
            row.model,
            row.initial_stub_transition,
            row.regular_transition,
            row.key,
            path,
            row.exercise_count,
            paths_per_price,
            row_spots + path
        );
        const float terminal_spot = expf(terminal.log_spot);
        row_cashflows[path] = immediate_payoff<Side>(terminal_spot, row.strike);
    }
}

// Produce one deterministic normal-equation partial per path block.
template <OptionSide Side>
__global__ void regression_partials_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const float* __restrict__ observed_spots,
    const float* __restrict__ cashflows,
    double* __restrict__ regression_partials
) {
    __shared__ PreparedRow row;
    if (threadIdx.x == 0U) row = prepared_rows[blockIdx.y];
    __syncthreads();
    if (backward_level >= row.exercise_count) return;

    const std::size_t regression_exercise =
        static_cast<std::size_t>(row.exercise_count - 1U - backward_level);
    const std::size_t state_offset =
        row.state_offset + regression_exercise * paths_per_price;
    const float* const row_spots = observed_spots + state_offset;
    const float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    double values[RegressionBasis::kRegressionValueCount] = {};

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const float spot = row_spots[path];
        const float immediate = immediate_payoff<Side>(spot, row.strike);
        if (immediate <= 0.0f) continue;

        const RegressionBasis basis =
            RegressionBasis::evaluate(
                spot * row.inverse_strike,
                logf(fmaxf(spot * row.inverse_strike, 1.0e-6f))
            );
        const double target =
            static_cast<double>(row.exercise_discount)
            * static_cast<double>(row_cashflows[path]);
        lsm::accumulate_normal_equations(basis, target, values);
    }

    lsm::reduce_and_store_regression_partials(
        values,
        blockIdx.y,
        blockIdx.x,
        blocks_per_price,
        regression_partials
    );
}

// Reduce all path blocks, regularize G, and solve one system per price.
__global__ void solve_regressions_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t blocks_per_price,
    const double* __restrict__ regression_partials,
    double* __restrict__ regression_coefficients,
    std::uint32_t* __restrict__ regression_valid
) {
    const std::size_t batch_price = blockIdx.x;
    lsm::solve_regression_for_row(
        prepared_rows[batch_price].exercise_count,
        backward_level,
        batch_price,
        blocks_per_price,
        regression_partials,
        regression_coefficients,
        regression_valid
    );
}

// Apply one fitted continuation policy and update every discounted cashflow.
template <OptionSide Side>
__global__ void update_cashflows_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t paths_per_price,
    const float* __restrict__ observed_spots,
    const double* __restrict__ regression_coefficients,
    const std::uint32_t* __restrict__ regression_valid,
    float* __restrict__ cashflows
) {
    __shared__ PreparedRow row;
    __shared__ float coefficients[RegressionBasis::kSize];
    __shared__ std::uint32_t solved;
    if (threadIdx.x == 0U) {
        row = prepared_rows[blockIdx.y];
        solved = regression_valid[blockIdx.y];
        for (std::size_t index = 0U;
             index < RegressionBasis::kSize;
             ++index) {
            coefficients[index] = static_cast<float>(
                regression_coefficients[
                    blockIdx.y * RegressionBasis::kSize + index
                ]
            );
        }
    }
    __syncthreads();
    if (backward_level >= row.exercise_count) return;

    const std::size_t regression_exercise =
        static_cast<std::size_t>(row.exercise_count - 1U - backward_level);
    const std::size_t state_offset =
        row.state_offset + regression_exercise * paths_per_price;
    const float* const row_spots = observed_spots + state_offset;
    float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const float spot = row_spots[path];
        const float immediate = immediate_payoff<Side>(spot, row.strike);
        float updated = row.exercise_discount * row_cashflows[path];

        if (solved != 0U && immediate > 0.0f) {
            const RegressionBasis basis =
                RegressionBasis::evaluate(
                    spot * row.inverse_strike,
                    logf(fmaxf(spot * row.inverse_strike, 1.0e-6f))
                );
            float continuation = 0.0f;
            #pragma unroll
            for (std::size_t index = 0U;
                 index < RegressionBasis::kSize;
                 ++index) {
                continuation = fmaf(
                    coefficients[index],
                    basis.values[index],
                    continuation
                );
            }
            if (immediate > continuation) {
                updated = immediate;
            }
        }
        row_cashflows[path] = updated;
    }
}

// Produce one FP64 payoff-moment partial per path block.
__global__ void moment_partials_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const float* __restrict__ cashflows,
    double* __restrict__ moment_partials
) {
    const float initial_discount =
        prepared_rows[blockIdx.y].initial_discount;
    const float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    double sum = 0.0;
    double sumsq = 0.0;

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const float value = initial_discount * row_cashflows[path];
        const double accumulated_value = static_cast<double>(value);
        sum += accumulated_value;
        sumsq += accumulated_value * accumulated_value;
    }

    const reductions::MomentSums totals =
        reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        const std::size_t batch_price = blockIdx.y;
        moment_partials[
            (batch_price * kMomentValueCount) * blocks_per_price + blockIdx.x
        ] = totals.sum;
        moment_partials[
            (batch_price * kMomentValueCount + 1U) * blocks_per_price
            + blockIdx.x
        ] = totals.sumsq;
    }
}

// Finish moments and compare continuation with deterministic time-zero exercise.
template <OptionSide Side>
__global__ void finalize_prices_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const double* __restrict__ moment_partials,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    const std::size_t batch_price = blockIdx.x;
    const double* const row_sums =
        moment_partials
        + (batch_price * kMomentValueCount) * blocks_per_price;
    const double* const row_sumsq =
        moment_partials
        + (batch_price * kMomentValueCount + 1U) * blocks_per_price;
    double sum = 0.0;
    double sumsq = 0.0;

    for (std::size_t partial = threadIdx.x;
         partial < blocks_per_price;
         partial += blockDim.x) {
        sum += row_sums[partial];
        sumsq += row_sumsq[partial];
    }
    const reductions::MomentSums totals =
        reductions::reduce_block(sum, sumsq);
    if (threadIdx.x != 0U) return;

    double continuation = 0.0;
    double standard_error = 0.0;
    reductions::compute_statistics(
        totals, paths_per_price, continuation, standard_error
    );
    const PreparedRow* const row = prepared_rows + batch_price;
    const double immediate = static_cast<double>(
        immediate_payoff<Side>(row->initial_spot, row->strike)
    );
    if (immediate > continuation) {
        prices[row->result_index] = static_cast<float>(immediate);
        standard_errors[row->result_index] = 0.0f;
    } else {
        prices[row->result_index] = static_cast<float>(continuation);
        standard_errors[row->result_index] =
            static_cast<float>(standard_error);
    }
}

// Validate pointers, construction, Monte Carlo dimensions, and the 2D grid.
void validate_variance_gamma_american_option_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t paths_per_price,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(device_standard_errors, "device_standard_errors");
    if (host_products == nullptr) {
        throw std::invalid_argument("host_products is null.");
    }
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    validate_monte_carlo_path_count(paths_per_price);
    validate_day_fraction(day_fraction);
    validate_reduction_block_size(threads_per_block);
    validate_row_seed_range(result_count, base_seed);

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );
    if (blocks_per_price == 0U
        || blocks_per_price
            > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::invalid_argument(
            "blocks_per_price must fit the current gridDim.x limit."
        );
    }
}

}  // namespace

// Price every row with one persistent workspace and memory-aware batches.
template <OptionSide Side>
lsm::LaunchResult launch_variance_gamma_american_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    validate_variance_gamma_american_option_launch(
        device_models,
        model_count,
        host_products,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        monte_carlo_paths_per_price,
        day_fraction,
        threads_per_block,
        blocks_per_price,
        base_seed,
        device_prices,
        device_standard_errors
    );
    const std::size_t path_block_capacity =
        1U + (monte_carlo_paths_per_price - 1U) / threads_per_block;
    const std::size_t launched_blocks_per_price = std::min(
        blocks_per_price, path_block_capacity
    );
    const char* const name = product_name<Side>();
    const lsm::WorkspaceBudget budget = lsm::query_workspace_budget(name);
    const lsm::WorkspaceDescriptor descriptor = workspace_descriptor();
    const std::vector<lsm::EarlyExerciseRowPlan> row_plans = make_row_plans(
        host_products,
        product_count,
        cartesian_product,
        result_count,
        monte_carlo_paths_per_price,
        name
    );
    const lsm::ExecutionPlan plan = lsm::plan_batches(
        row_plans,
        descriptor,
        monte_carlo_paths_per_price,
        launched_blocks_per_price,
        budget.available_bytes,
        name
    );
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );
    if (plan.maximum_prices_per_batch
        > static_cast<std::size_t>(properties.maxGridSize[1])) {
        throw std::overflow_error(
            std::string(name) + " batch exceeds the current gridDim.y limit."
        );
    }

    lsm::LaunchResources resources(plan.maximum_workspace_bytes, name);
    double kernel_seconds = 0.0;
    std::size_t kernel_launch_count = 0U;
    std::vector<std::uint32_t> host_exercise_counts(
        plan.maximum_prices_per_batch
    );
    std::vector<std::size_t> host_state_offsets(
        plan.maximum_prices_per_batch
    );

    for (const lsm::BatchPlan& batch : plan.batches) {
        std::size_t state_value_cursor = 0U;
        for (std::size_t batch_price = 0U;
             batch_price < batch.result_count;
             ++batch_price) {
            const lsm::EarlyExerciseRowPlan& row_plan =
                row_plans[batch.result_offset + batch_price];
            host_exercise_counts[batch_price] = row_plan.exercise_count;
            host_state_offsets[batch_price] = state_value_cursor;
            state_value_cursor += row_plan.state_value_count;
        }
        if (state_value_cursor != batch.state_value_count) {
            throw std::logic_error(
                std::string(name)
                + " batch state count changed after planning."
            );
        }

        const lsm::WorkspaceLayout layout = lsm::make_workspace_layout(
            descriptor,
            batch.result_count,
            batch.state_value_count,
            monte_carlo_paths_per_price,
            launched_blocks_per_price,
            name
        );
        const StateRegions states = variance_gamma_state_regions(layout);
        unsigned char* const device_workspace = resources.workspace();
        PreparedRow* const prepared_rows =
            lsm::workspace_pointer<PreparedRow>(
                device_workspace, layout.prepared_rows
            );
        std::uint32_t* const device_exercise_counts =
            lsm::workspace_pointer<std::uint32_t>(
                device_workspace, layout.exercise_counts
            );
        std::size_t* const device_state_offsets =
            lsm::workspace_pointer<std::size_t>(
                device_workspace, layout.state_offsets
            );
        float* const observed_spots = lsm::workspace_pointer<float>(
            device_workspace, states.spots
        );
        float* const cashflows = lsm::workspace_pointer<float>(
            device_workspace, layout.cashflows
        );
        double* const regression_partials = lsm::workspace_pointer<double>(
            device_workspace, layout.regression_partials
        );
        double* const regression_coefficients = lsm::workspace_pointer<double>(
            device_workspace, layout.regression_coefficients
        );
        std::uint32_t* const regression_valid =
            lsm::workspace_pointer<std::uint32_t>(
                device_workspace, layout.regression_valid
            );
        double* const moment_partials = lsm::workspace_pointer<double>(
            device_workspace, layout.moment_partials
        );

        check_cuda(
            cudaMemcpy(
                device_exercise_counts,
                host_exercise_counts.data(),
                batch.result_count * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy American-option exercise counts"
        );
        check_cuda(
            cudaMemcpy(
                device_state_offsets,
                host_state_offsets.data(),
                batch.result_count * sizeof(std::size_t),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy American-option state offsets"
        );

        const dim3 path_grid(
            static_cast<unsigned int>(launched_blocks_per_price),
            static_cast<unsigned int>(batch.result_count),
            1U
        );
        const unsigned int row_blocks = static_cast<unsigned int>(
            (batch.result_count + threads_per_block - 1U)
            / threads_per_block
        );
        const std::size_t warp_count = threads_per_block / 32U;
        const std::size_t regression_shared_bytes =
            lsm::regression_shared_bytes(threads_per_block);
        const std::size_t moment_shared_bytes =
            2U * warp_count * sizeof(double);

        resources.start_batch();

        report_cuda_kernel_launch_if_enabled(
            "variance_gamma.american_option.prepare_rows",
            "common",
            prepare_rows_kernel,
            dim3(row_blocks),
            dim3(threads_per_block)
        );
        prepare_rows_kernel<<<row_blocks, threads_per_block>>>(
                device_models,
                device_products,
                product_count,
                cartesian_product,
                batch.result_count,
                day_fraction,
                base_seed,
                batch.result_offset,
                device_exercise_counts,
                device_state_offsets,
                prepared_rows
            );
        check_cuda(cudaGetLastError(), "prepare American-option rows");
        ++kernel_launch_count;

        report_cuda_kernel_launch_if_enabled(
            "variance_gamma.american_option.simulate_paths",
            option_side_name(Side),
            simulate_paths_kernel<Side>,
            path_grid,
            dim3(threads_per_block)
        );
        simulate_paths_kernel<Side><<<path_grid, threads_per_block>>>(
                prepared_rows,
                monte_carlo_paths_per_price,
                observed_spots,
                cashflows
            );
        check_cuda(cudaGetLastError(), "simulate American-option paths");
        ++kernel_launch_count;

        for (std::uint32_t backward_level = 1U;
             backward_level < batch.maximum_exercise_count;
             ++backward_level) {
            report_cuda_kernel_launch_if_enabled(
                "variance_gamma.american_option.regression_partials",
                option_side_name(Side),
                regression_partials_kernel<Side>,
                path_grid,
                dim3(threads_per_block),
                regression_shared_bytes
            );
            regression_partials_kernel<Side><<<
                    path_grid,
                    threads_per_block,
                    regression_shared_bytes
                >>>(
                    prepared_rows,
                    backward_level,
                    monte_carlo_paths_per_price,
                    launched_blocks_per_price,
                    observed_spots,
                    cashflows,
                    regression_partials
                );
            check_cuda(
                cudaGetLastError(), "American-option regression partials"
            );

            report_cuda_kernel_launch_if_enabled(
                "variance_gamma.american_option.solve_regressions",
                "common",
                solve_regressions_kernel,
                dim3(static_cast<unsigned int>(batch.result_count)),
                dim3(threads_per_block),
                regression_shared_bytes
            );
            solve_regressions_kernel<<<
                    static_cast<unsigned int>(batch.result_count),
                    threads_per_block,
                    regression_shared_bytes
                >>>(
                    prepared_rows,
                    backward_level,
                    launched_blocks_per_price,
                    regression_partials,
                    regression_coefficients,
                    regression_valid
                );
            check_cuda(
                cudaGetLastError(), "solve American-option regressions"
            );

            report_cuda_kernel_launch_if_enabled(
                "variance_gamma.american_option.update_cashflows",
                option_side_name(Side),
                update_cashflows_kernel<Side>,
                path_grid,
                dim3(threads_per_block)
            );
            update_cashflows_kernel<Side><<<path_grid, threads_per_block>>>(
                    prepared_rows,
                    backward_level,
                    monte_carlo_paths_per_price,
                    observed_spots,
                    regression_coefficients,
                    regression_valid,
                    cashflows
                );
            check_cuda(
                cudaGetLastError(), "update American-option cashflows"
            );
            kernel_launch_count += 3U;
        }

        report_cuda_kernel_launch_if_enabled(
            "variance_gamma.american_option.moment_partials",
            "common",
            moment_partials_kernel,
            path_grid,
            dim3(threads_per_block),
            moment_shared_bytes
        );
        moment_partials_kernel<<<
                path_grid,
                threads_per_block,
                moment_shared_bytes
            >>>(
                prepared_rows,
                monte_carlo_paths_per_price,
                launched_blocks_per_price,
                cashflows,
                moment_partials
            );
        check_cuda(cudaGetLastError(), "American-option moment partials");
        ++kernel_launch_count;

        report_cuda_kernel_launch_if_enabled(
            "variance_gamma.american_option.finalize_prices",
            option_side_name(Side),
            finalize_prices_kernel<Side>,
            dim3(static_cast<unsigned int>(batch.result_count)),
            dim3(threads_per_block),
            moment_shared_bytes
        );
        finalize_prices_kernel<Side><<<
                static_cast<unsigned int>(batch.result_count),
                threads_per_block,
                moment_shared_bytes
            >>>(
                prepared_rows,
                monte_carlo_paths_per_price,
                launched_blocks_per_price,
                moment_partials,
                device_prices,
                device_standard_errors
            );
        check_cuda(cudaGetLastError(), "finalize American-option prices");
        ++kernel_launch_count;
        kernel_seconds += resources.finish_batch();
    }

    return {
        kernel_seconds,
        plan.batches.size(),
        kernel_launch_count,
        plan.maximum_prices_per_batch,
        launched_blocks_per_price,
        plan.maximum_workspace_bytes,
    };
}

// Build both public payoff specializations in this CUDA translation unit.
using LaunchSignature =
    decltype(launch_variance_gamma_american_option_cuda<OptionSide::call>);
template LaunchSignature
launch_variance_gamma_american_option_cuda<OptionSide::call>;
template LaunchSignature
launch_variance_gamma_american_option_cuda<OptionSide::put>;

}  // namespace ai_factory::workbench::variance_gamma
