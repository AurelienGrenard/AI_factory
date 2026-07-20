#include "ai_factory/cuda/rough_bergomi/api.cuh"
#include "ai_factory/cuda/rough_bergomi/dynamics.cuh"
#include "ai_factory/cuda/common/autocall_reduction.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace ai_factory::cuda {
namespace {

constexpr int kThreads = 128;
struct RoughBergomiAutocallWorkspaceTag {};

template <std::size_t MaxSteps>
__global__ void rough_bergomi_autocall_partial_kernel(
    const RoughBergomiAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partials
) {
    const auto row_index =
        static_cast<std::size_t>(blockIdx.x) / path_blocks_per_row;
    const auto path_block =
        static_cast<std::size_t>(blockIdx.x) % path_blocks_per_row;
    if (row_index >= row_count) {
        return;
    }
    const auto row = rows[row_index];
    const auto path = path_block * static_cast<std::size_t>(blockDim.x)
                      + static_cast<std::size_t>(threadIdx.x);
    const double dt = row.model.maturity / static_cast<double>(num_steps);
    __shared__ double weights[MaxSteps + 1U];
    __shared__ double variance_time_powers[MaxSteps];
    for (std::size_t step = threadIdx.x; step < num_steps; step += blockDim.x) {
        weights[step] = step >= 2U
                            ? rough_bergomi_detail::rough_bergomi_optimal_weight(
                                  row.model.alpha, dt, step
                              )
                            : 0.0;
        const double time = static_cast<double>(step) * dt;
        variance_time_powers[step] =
            pow(time, 2.0 * row.model.alpha + 1.0);
    }
    __syncthreads();

    autocall_detail::PathMetrics metrics{};
    if (path < num_paths) {
        double dws[MaxSteps];
        const double sqrt_dt = sqrt(dt);
        rough_bergomi_detail::prepare_rough_bergomi_path_normals<MaxSteps>(
            row.model, num_steps, path, sqrt_dt, dws
        );
        const auto first_index = path * num_steps;
        rng::NormalSequence residual_normals(row.model.seed, 1U, first_index);
        rng::NormalSequence stock_normals(row.model.seed, 2U, first_index);
        const double drift =
            (row.model.risk_free_rate - row.model.dividend_yield) * dt;
        const double rho_perp = sqrt(1.0 - row.model.rho * row.model.rho);
        const auto observation_count = row.product.observation_count;
        const auto stride = num_steps / observation_count;
        autocall_detail::PathState state{};
        double log_spot = log(row.model.spot);
        double spot = row.model.spot;
        std::size_t observation = 0U;
        std::size_t call_observation = 0U;
        for (std::size_t step = 0U; step < num_steps; ++step) {
            const double y = rough_bergomi_detail::rough_bergomi_volterra_state<MaxSteps>(
                row.model,
                num_steps,
                path,
                step,
                dt,
                sqrt_dt,
                step == 0U ? 0.0 : residual_normals.next(),
                weights,
                dws
            );
            const double variance = rough_bergomi_detail::rough_bergomi_variance(
                row.model, y, variance_time_powers[step]
            );
            const double dz = row.model.rho * dws[step]
                              + rho_perp * sqrt_dt * stock_normals.next();
            log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
            spot = exp(log_spot);
            if ((step + 1U) % stride == 0U) {
                ++observation;
                const double time = row.model.maturity
                                    * static_cast<double>(observation)
                                    / static_cast<double>(observation_count);
                if (autocall_detail::observe(
                        row.product,
                        spot / row.model.spot,
                        observation,
                        time,
                        row.model.risk_free_rate,
                        state
                    )) {
                    call_observation = observation;
                    break;
                }
            }
        }
        metrics = autocall_detail::finish(
            row.product,
            spot / row.model.spot,
            row.model.maturity,
            row.model.risk_free_rate,
            call_observation,
            state
        );
    }
    double values[autocall_detail::kMetricCount]{};
    autocall_detail::metrics_to_values(metrics, values);
    const auto partial_count = row_count * path_blocks_per_row;
    autocall_detail::reduce_and_store(
        values,
        row_index * path_blocks_per_row + path_block,
        partial_count,
        partials
    );
}

template <std::size_t MaxSteps>
void launch_partial(
    const RoughBergomiAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partials,
    std::size_t shared_bytes
) {
    rough_bergomi_autocall_partial_kernel<MaxSteps><<<
        static_cast<unsigned int>(row_count * path_blocks_per_row),
        kThreads,
        shared_bytes
    >>>(
        rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        partials
    );
}

void dispatch_partial(
    const RoughBergomiAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partials,
    std::size_t shared_bytes
) {
    if (num_steps <= 128U) {
        launch_partial<128U>(rows, row_count, num_paths, num_steps, path_blocks_per_row, partials, shared_bytes);
    } else if (num_steps <= 160U) {
        launch_partial<160U>(rows, row_count, num_paths, num_steps, path_blocks_per_row, partials, shared_bytes);
    } else if (num_steps <= 208U) {
        launch_partial<208U>(rows, row_count, num_paths, num_steps, path_blocks_per_row, partials, shared_bytes);
    } else if (num_steps <= rough_bergomi_detail::kMaxRoughBergomiSteps) {
        launch_partial<rough_bergomi_detail::kMaxRoughBergomiSteps>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row, partials, shared_bytes
        );
    } else {
        throw std::invalid_argument(
            "Optimized rough Bergomi autocall kernels support at most 272 steps."
        );
    }
}

}  // namespace

void price_rough_bergomi_autocall_cuda(
    const RoughBergomiAutocallRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    AutocallOutput* host_outputs,
    CudaTiming* timing
) {
    if (row_count == 0U || num_paths < 2U) {
        throw std::invalid_argument("Autocall pricing requires rows and at least two paths.");
    }
    for (std::size_t index = 0; index < row_count; ++index) {
        if (host_rows[index].product.observation_count == 0U
            || num_steps % host_rows[index].product.observation_count != 0U) {
            throw std::invalid_argument(
                "Autocall observation count must divide the simulation step count."
            );
        }
    }
    const auto path_blocks_per_row =
        (num_paths + static_cast<std::size_t>(kThreads) - 1U)
        / static_cast<std::size_t>(kThreads);
    const auto partial_count = row_count * path_blocks_per_row;
    auto& workspace = detail::reusable_cuda_workspace<
        RoughBergomiAutocallWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<RoughBergomiAutocallRow>(
        0U, row_count, "cudaMalloc rough Bergomi autocall rows"
    );
    auto* device_outputs = workspace.buffer<AutocallOutput>(
        1U, row_count, "cudaMalloc rough Bergomi autocall outputs"
    );
    auto* partials = workspace.buffer<double>(
        2U,
        autocall_detail::kMetricCount * partial_count,
        "cudaMalloc rough Bergomi autocall partials"
    );
    detail::check_cuda(
        cudaMemcpy(
            device_rows,
            host_rows,
            row_count * sizeof(RoughBergomiAutocallRow),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy rough Bergomi autocall rows"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start), "cudaEventRecord rough autocall start");
    const auto reduction_shared_bytes =
        autocall_detail::kMetricCount * kThreads * sizeof(double);
    const auto model_shared_bytes =
        (rough_bergomi_detail::kMaxRoughBergomiSteps + 1U
         + rough_bergomi_detail::kMaxRoughBergomiSteps)
        * sizeof(double);
    const auto shared_bytes =
        model_shared_bytes > reduction_shared_bytes
            ? model_shared_bytes
            : reduction_shared_bytes;
    dispatch_partial(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        partials,
        shared_bytes
    );
    detail::check_cuda(cudaGetLastError(), "rough Bergomi autocall partial kernel");
    autocall_detail::autocall_finalize_kernel<<<
        static_cast<unsigned int>(row_count),
        kThreads,
        reduction_shared_bytes
    >>>(
        partials,
        row_count,
        path_blocks_per_row,
        num_paths,
        device_outputs
    );
    detail::check_cuda(cudaGetLastError(), "rough Bergomi autocall finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "cudaEventRecord rough autocall stop");
    detail::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize rough autocall");
    float elapsed_ms = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "rough autocall timing");
    detail::check_cuda(
        cudaMemcpy(
            host_outputs,
            device_outputs,
            row_count * sizeof(AutocallOutput),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy rough Bergomi autocall outputs"
    );
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
}

}  // namespace ai_factory::cuda
