#include "ai_factory/cuda/black_scholes/api.cuh"
#include "ai_factory/cuda/black_scholes/dynamics.cuh"
#include "ai_factory/cuda/common/autocall_reduction.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace ai_factory::cuda {
namespace {

constexpr int kThreads = 256;
struct BlackScholesAutocallWorkspaceTag {};

__global__ void black_scholes_autocall_partial_kernel(
    const BlackScholesAutocallRow* rows,
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
    autocall_detail::PathMetrics metrics{};
    if (path < num_paths) {
        const auto observation_count = row.product.observation_count;
        const auto stride = num_steps / observation_count;
        const double dt = row.model.maturity / static_cast<double>(num_steps);
        const double sqrt_dt = sqrt(dt);
        rng::NormalSequence normals(row.model.seed, 0U, path * num_steps);
        autocall_detail::PathState state{};
        double spot = row.model.spot;
        std::size_t observation = 0U;
        std::size_t call_observation = 0U;
        for (std::size_t step = 1U; step <= num_steps; ++step) {
            spot *= exp(black_scholes_detail::log_step(
                row.model, dt, sqrt_dt, normals.next()
            ));
            if (step % stride == 0U) {
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
    const auto partial_index = row_index * path_blocks_per_row + path_block;
    autocall_detail::reduce_and_store(
        values, partial_index, partial_count, partials
    );
}

}  // namespace

void price_black_scholes_autocall_cuda(
    const BlackScholesAutocallRow* host_rows,
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
        BlackScholesAutocallWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<BlackScholesAutocallRow>(
        0U, row_count, "cudaMalloc Black-Scholes autocall rows"
    );
    auto* device_outputs = workspace.buffer<AutocallOutput>(
        1U, row_count, "cudaMalloc Black-Scholes autocall outputs"
    );
    auto* partials = workspace.buffer<double>(
        2U,
        autocall_detail::kMetricCount * partial_count,
        "cudaMalloc Black-Scholes autocall partials"
    );
    detail::check_cuda(
        cudaMemcpy(
            device_rows,
            host_rows,
            row_count * sizeof(BlackScholesAutocallRow),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy Black-Scholes autocall rows"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start), "cudaEventRecord autocall start");
    const auto shared_bytes =
        autocall_detail::kMetricCount * kThreads * sizeof(double);
    black_scholes_autocall_partial_kernel<<<
        static_cast<unsigned int>(partial_count), kThreads, shared_bytes
    >>>(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        partials
    );
    detail::check_cuda(cudaGetLastError(), "Black-Scholes autocall partial kernel");
    autocall_detail::autocall_finalize_kernel<<<
        static_cast<unsigned int>(row_count), kThreads, shared_bytes
    >>>(
        partials,
        row_count,
        path_blocks_per_row,
        num_paths,
        device_outputs
    );
    detail::check_cuda(cudaGetLastError(), "Black-Scholes autocall finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "cudaEventRecord autocall stop");
    detail::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize autocall");
    float elapsed_ms = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "autocall timing");
    detail::check_cuda(
        cudaMemcpy(
            host_outputs,
            device_outputs,
            row_count * sizeof(AutocallOutput),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy Black-Scholes autocall outputs"
    );
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
}

}  // namespace ai_factory::cuda
