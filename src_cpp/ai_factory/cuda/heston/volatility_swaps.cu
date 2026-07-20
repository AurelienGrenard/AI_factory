#include "ai_factory/cuda/heston/api.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::cuda {
namespace {

using rng::standard_normal;
using rng::standard_normal_pair;
using reductions::reduce_block;
using detail::check_cuda;
constexpr int kThreadsPerBlock = 128;
using detail::reusable_cuda_workspace;
using heston_detail::advance_heston_qe_step_precomputed;
using heston_detail::advance_heston_step;
using heston_detail::kHestonEulerFullTruncation;
using heston_detail::make_qe_coefficients;

constexpr double kObservationsPerYear = 52.0;
struct HestonVolatilityWorkspaceTag {};

__device__ __forceinline__ double simulate_heston_euler_realized_volatility(
    const HestonRow& row,
    std::size_t num_steps,
    std::size_t path,
    double dt,
    double sqrt_dt,
    double drift_scale,
    double correlation_scale,
    double kappa_dt,
    double vol_of_var_sqrt_dt
) {
    double spot = row.spot;
    double variance = row.initial_variance;
    double sum_squared_log_returns = 0.0;
    std::size_t step = 0U;
    for (; step + 1U < num_steps;) {
        const auto normal_index = path * num_steps + step;
        if ((normal_index & 1ULL) != 0ULL) {
            break;
        }
        const auto pair_index = normal_index / 2ULL;
        const auto spot_pair = standard_normal_pair(row.seed, 0U, pair_index);
        const auto variance_pair = standard_normal_pair(row.seed, 1U, pair_index);

        double previous_spot = spot;
        advance_heston_step(
            row,
            dt,
            sqrt_dt,
            drift_scale,
            correlation_scale,
            kappa_dt,
            vol_of_var_sqrt_dt,
            spot_pair.first,
            variance_pair.first,
            spot,
            variance
        );
        double log_return = log(spot / previous_spot);
        sum_squared_log_returns += log_return * log_return;

        previous_spot = spot;
        advance_heston_step(
            row,
            dt,
            sqrt_dt,
            drift_scale,
            correlation_scale,
            kappa_dt,
            vol_of_var_sqrt_dt,
            spot_pair.second,
            variance_pair.second,
            spot,
            variance
        );
        log_return = log(spot / previous_spot);
        sum_squared_log_returns += log_return * log_return;
        step += 2U;
    }
    for (; step < num_steps; ++step) {
        const auto normal_index = path * num_steps + step;
        const double previous_spot = spot;
        advance_heston_step(
            row,
            dt,
            sqrt_dt,
            drift_scale,
            correlation_scale,
            kappa_dt,
            vol_of_var_sqrt_dt,
            standard_normal(row.seed, 0U, normal_index),
            standard_normal(row.seed, 1U, normal_index),
            spot,
            variance
        );
        const double log_return = log(spot / previous_spot);
        sum_squared_log_returns += log_return * log_return;
    }
    return sqrt(kObservationsPerYear / static_cast<double>(num_steps)
                * sum_squared_log_returns);
}

__device__ __forceinline__ double simulate_heston_qe_realized_volatility(
    const HestonRow& row,
    std::size_t num_steps,
    std::size_t path,
    double dt
) {
    double log_spot = log(row.spot);
    double variance = row.initial_variance;
    double sum_squared_log_returns = 0.0;
    const auto coefficients = make_qe_coefficients(row, dt);
    for (std::size_t step = 0U; step < num_steps; ++step) {
        const double previous_log_spot = log_spot;
        const auto normal_index = path * num_steps + step;
        advance_heston_qe_step_precomputed(
            row,
            coefficients,
            standard_normal(row.seed, 0U, normal_index),
            rng::standard_uniform(row.seed, 2U, normal_index),
            standard_normal(row.seed, 1U, normal_index),
            log_spot,
            variance
        );
        const double log_return = log_spot - previous_log_spot;
        sum_squared_log_returns += log_return * log_return;
    }
    return sqrt(kObservationsPerYear / static_cast<double>(num_steps)
                * sum_squared_log_returns);
}

__device__ __forceinline__ double simulate_heston_realized_volatility(
    const HestonRow& row,
    std::size_t num_steps,
    std::size_t path,
    double dt,
    double sqrt_dt,
    double drift_scale,
    double correlation_scale,
    double kappa_dt,
    double vol_of_var_sqrt_dt
) {
    if (row.scheme == kHestonEulerFullTruncation) {
        return simulate_heston_euler_realized_volatility(
            row,
            num_steps,
            path,
            dt,
            sqrt_dt,
            drift_scale,
            correlation_scale,
            kappa_dt,
            vol_of_var_sqrt_dt
        );
    }
    return simulate_heston_qe_realized_volatility(row, num_steps, path, dt);
}

__global__ void heston_volatility_swap_partial_kernel(
    const HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partial_sums,
    double* partial_sumsq
) {
    const auto row_index =
        static_cast<std::size_t>(blockIdx.x) / path_blocks_per_row;
    const auto path_block =
        static_cast<std::size_t>(blockIdx.x) % path_blocks_per_row;
    if (row_index >= row_count) {
        return;
    }

    const auto row = rows[row_index];
    const auto path =
        path_block * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift_scale = row.risk_free_rate - row.dividend_yield;
    const double correlation_scale = sqrt(1.0 - row.rho * row.rho);
    const double kappa_dt = row.kappa * dt;
    const double vol_of_var_sqrt_dt = row.volatility_of_variance * sqrt_dt;
    const double discount = exp(-row.risk_free_rate * row.maturity);
    double local_sum = 0.0;
    double local_sumsq = 0.0;

    if (path < num_paths) {
        const double realized_volatility = simulate_heston_realized_volatility(
            row,
            num_steps,
            path,
            dt,
            sqrt_dt,
            drift_scale,
            correlation_scale,
            kappa_dt,
            vol_of_var_sqrt_dt
        );
        const double payoff = discount * (realized_volatility - row.strike);
        local_sum = payoff;
        local_sumsq = payoff * payoff;
    }

    reduce_block(local_sum, local_sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto partial_index = row_index * path_blocks_per_row + path_block;
        partial_sums[partial_index] = shared[0];
        partial_sumsq[partial_index] = shared[blockDim.x];
    }
}

__global__ void heston_volatility_swap_finalize_kernel(
    const double* partial_sums,
    const double* partial_sumsq,
    std::size_t row_count,
    std::size_t path_blocks_per_row,
    std::size_t num_paths,
    MonteCarloOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (std::size_t block = threadIdx.x; block < path_blocks_per_row; block += blockDim.x) {
        const auto partial_index = row_index * path_blocks_per_row + block;
        local_sum += partial_sums[partial_index];
        local_sumsq += partial_sumsq[partial_index];
    }

    reduce_block(local_sum, local_sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const double path_count = static_cast<double>(num_paths);
        const double mean = shared[0] / path_count;
        const double variance =
            (shared[blockDim.x] - path_count * mean * mean)
            / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {
            mean,
            sqrt(fmax(variance, 0.0)) / sqrt(path_count),
        };
    }
}

}  // namespace

void price_heston_volatility_swap_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    const auto row_bytes = row_count * sizeof(HestonRow);
    const auto output_bytes = row_count * sizeof(MonteCarloOutput);
    const auto path_blocks_per_row =
        (num_paths + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    const auto partial_count = row_count * path_blocks_per_row;
    const auto shared_bytes = 2U * kThreadsPerBlock * sizeof(double);

    auto& workspace = reusable_cuda_workspace<HestonVolatilityWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<HestonRow>(0U, row_count, "cudaMalloc rows");
    auto* device_outputs = workspace.buffer<MonteCarloOutput>(
        1U, row_count, "cudaMalloc outputs"
    );
    auto* device_partials = workspace.buffer<double>(
        2U, 2U * partial_count, "cudaMalloc partials"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    double* const device_partial_sums = device_partials;
    double* const device_partial_sumsq = device_partials + partial_count;
    try {
        check_cuda(
            cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy rows"
        );
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");

        const auto partial_block_count =
            static_cast<unsigned int>(row_count * path_blocks_per_row);
        heston_volatility_swap_partial_kernel<<<
            partial_block_count,
            kThreadsPerBlock,
            shared_bytes
        >>>(
            device_rows,
            row_count,
            num_paths,
            num_steps,
            path_blocks_per_row,
            device_partial_sums,
            device_partial_sumsq
        );
        check_cuda(cudaGetLastError(), "heston volatility swap partial kernel launch");

        heston_volatility_swap_finalize_kernel<<<
            static_cast<unsigned int>(row_count),
            kThreadsPerBlock,
            shared_bytes
        >>>(
            device_partial_sums,
            device_partial_sumsq,
            row_count,
            path_blocks_per_row,
            num_paths,
            device_outputs
        );
        check_cuda(cudaGetLastError(), "heston volatility swap finalize kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing");
        check_cuda(
            cudaMemcpy(host_outputs, device_outputs, output_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy outputs"
        );
        if (timing != nullptr) {
            timing->simulation_ms = elapsed_ms;
            timing->total_ms = elapsed_ms;
        }
    } catch (...) {
        throw;
    }
}

}  // namespace ai_factory::cuda
