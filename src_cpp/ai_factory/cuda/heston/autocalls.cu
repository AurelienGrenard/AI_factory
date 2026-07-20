#include "ai_factory/cuda/heston/api.cuh"
#include "ai_factory/cuda/heston/dynamics.cuh"
#include "ai_factory/cuda/common/autocall_reduction.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace ai_factory::cuda {
namespace {

constexpr int kThreads = 128;
struct HestonAutocallWorkspaceTag {};

__global__ void heston_autocall_partial_kernel(
    const HestonAutocallRow* rows,
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
        const auto qe_coefficients = heston_detail::make_qe_coefficients(row.model, dt);
        const double correlation_scale = sqrt(1.0 - row.model.rho * row.model.rho);
        const double drift_scale = row.model.risk_free_rate - row.model.dividend_yield;
        const double kappa_dt = row.model.kappa * dt;
        const double vol_of_var_sqrt_dt = row.model.volatility_of_variance * sqrt_dt;
        autocall_detail::PathState state{};
        double spot = row.model.spot;
        double log_spot = log(row.model.spot);
        double variance = row.model.initial_variance;
        std::size_t observation = 0U;
        std::size_t call_observation = 0U;
        const auto first_index = path * num_steps;
        for (std::size_t step = 0U; step < num_steps; ++step) {
            const auto random_index = first_index + step;
            if (row.model.scheme == heston_detail::kHestonEulerFullTruncation) {
                heston_detail::advance_heston_step(
                    row.model,
                    dt,
                    sqrt_dt,
                    drift_scale,
                    correlation_scale,
                    kappa_dt,
                    vol_of_var_sqrt_dt,
                    rng::standard_normal(row.model.seed, 0U, random_index),
                    rng::standard_normal(row.model.seed, 1U, random_index),
                    spot,
                    variance
                );
            } else {
                heston_detail::advance_heston_qe_step_precomputed(
                    row.model,
                    qe_coefficients,
                    rng::standard_normal(row.model.seed, 0U, random_index),
                    rng::standard_uniform(row.model.seed, 2U, random_index),
                    rng::standard_normal(row.model.seed, 1U, random_index),
                    log_spot,
                    variance
                );
                spot = exp(log_spot);
            }
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

}  // namespace

void price_heston_autocall_cuda(
    const HestonAutocallRow* host_rows,
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
    auto& workspace =
        detail::reusable_cuda_workspace<HestonAutocallWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<HestonAutocallRow>(
        0U, row_count, "cudaMalloc Heston autocall rows"
    );
    auto* device_outputs = workspace.buffer<AutocallOutput>(
        1U, row_count, "cudaMalloc Heston autocall outputs"
    );
    auto* partials = workspace.buffer<double>(
        2U,
        autocall_detail::kMetricCount * partial_count,
        "cudaMalloc Heston autocall partials"
    );
    detail::check_cuda(
        cudaMemcpy(
            device_rows,
            host_rows,
            row_count * sizeof(HestonAutocallRow),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy Heston autocall rows"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start), "cudaEventRecord Heston autocall start");
    const auto shared_bytes =
        autocall_detail::kMetricCount * kThreads * sizeof(double);
    heston_autocall_partial_kernel<<<
        static_cast<unsigned int>(partial_count), kThreads, shared_bytes
    >>>(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        partials
    );
    detail::check_cuda(cudaGetLastError(), "Heston autocall partial kernel");
    autocall_detail::autocall_finalize_kernel<<<
        static_cast<unsigned int>(row_count), kThreads, shared_bytes
    >>>(
        partials,
        row_count,
        path_blocks_per_row,
        num_paths,
        device_outputs
    );
    detail::check_cuda(cudaGetLastError(), "Heston autocall finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "cudaEventRecord Heston autocall stop");
    detail::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize Heston autocall");
    float elapsed_ms = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "Heston autocall timing");
    detail::check_cuda(
        cudaMemcpy(
            host_outputs,
            device_outputs,
            row_count * sizeof(AutocallOutput),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy Heston autocall outputs"
    );
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
}

}  // namespace ai_factory::cuda
