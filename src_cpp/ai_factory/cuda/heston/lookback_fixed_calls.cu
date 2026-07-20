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
using rng::standard_uniform;
using reductions::reduce_block;
using reductions::reduce_block_four;
using detail::DeviceWorkspace;
using detail::allocate_workspace;
using detail::check_cuda;
constexpr int kThreadsPerBlock = 128;
using detail::release_workspace;
using detail::reusable_cuda_workspace;
using detail::run_kernel_common;
using detail::run_kernel_with_workspace;
using heston_detail::advance_heston_qe_step;
using heston_detail::advance_heston_step;
using heston_detail::kHestonAndersenQeMartingale;
using heston_detail::kHestonEulerFullTruncation;
using heston_detail::make_qe_coefficients;
using heston_detail::simulate_heston_qe_max_spot;

struct HestonLookbackWorkspaceTag {};

__global__ void heston_kernel(
    const HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto row = rows[row_index];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift_scale = row.risk_free_rate - row.dividend_yield;
    const double correlation_scale = sqrt(1.0 - row.rho * row.rho);
    const double kappa_dt = row.kappa * dt;
    const double vol_of_var_sqrt_dt = row.volatility_of_variance * sqrt_dt;
    const double discount = exp(-row.risk_free_rate * row.maturity);
    double local_sum = 0.0;
    double local_sumsq = 0.0;

    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        double spot = row.spot;
        double variance = row.initial_variance;
        if (row.scheme == kHestonEulerFullTruncation) {
            std::size_t step = 0;
            for (; step + 1U < num_steps;) {
                const auto normal_index = path * num_steps + step;
                if ((normal_index % 2ULL) != 0ULL) {
                    break;
                }
                const auto pair_index = normal_index / 2ULL;
                const auto spot_pair =
                    standard_normal_pair(row.seed, 0, pair_index);
                const auto variance_pair =
                    standard_normal_pair(row.seed, 1, pair_index);
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
                step += 2U;
            }
            for (; step < num_steps; ++step) {
                const auto normal_index = path * num_steps + step;
                advance_heston_step(
                    row,
                    dt,
                    sqrt_dt,
                    drift_scale,
                    correlation_scale,
                    kappa_dt,
                    vol_of_var_sqrt_dt,
                    standard_normal(row.seed, 0, normal_index),
                    standard_normal(row.seed, 1, normal_index),
                    spot,
                    variance
                );
            }
        } else {
            double log_spot = log(row.spot);
            for (std::size_t step = 0; step < num_steps; ++step) {
                const auto normal_index = path * num_steps + step;
                advance_heston_qe_step(
                    row,
                    dt,
                    standard_normal(row.seed, 0, normal_index),
                    standard_uniform(row.seed, 2, normal_index),
                    standard_normal(row.seed, 1, normal_index),
                    log_spot,
                    variance
                );
            }
            spot = exp(log_spot);
        }
        const double payoff = fmax(spot - row.strike, 0.0);
        const double discounted = discount * payoff;
        local_sum += discounted;
        local_sumsq += discounted * discounted;
    }

    reduce_block(local_sum, local_sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const double mean = shared[0] / static_cast<double>(num_paths);
        const double variance =
            (shared[blockDim.x] - static_cast<double>(num_paths) * mean * mean)
            / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {
            mean,
            sqrt(fmax(variance, 0.0)) / sqrt(static_cast<double>(num_paths)),
        };
    }
}

__global__ void heston_lookback_fixed_partial_kernel(
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
        double spot = row.spot;
        double max_spot = row.spot;
        double variance = row.initial_variance;
        if (row.scheme == kHestonEulerFullTruncation) {
            std::size_t step = 0;
            for (; step + 1U < num_steps;) {
                const auto normal_index = path * num_steps + step;
                if ((normal_index % 2ULL) != 0ULL) {
                    break;
                }
                const auto pair_index = normal_index / 2ULL;
                const auto spot_pair =
                    standard_normal_pair(row.seed, 0, pair_index);
                const auto variance_pair =
                    standard_normal_pair(row.seed, 1, pair_index);
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
                max_spot = fmax(max_spot, spot);
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
                max_spot = fmax(max_spot, spot);
                step += 2U;
            }
            for (; step < num_steps; ++step) {
                const auto normal_index = path * num_steps + step;
                advance_heston_step(
                    row,
                    dt,
                    sqrt_dt,
                    drift_scale,
                    correlation_scale,
                    kappa_dt,
                    vol_of_var_sqrt_dt,
                    standard_normal(row.seed, 0, normal_index),
                    standard_normal(row.seed, 1, normal_index),
                    spot,
                    variance
                );
                max_spot = fmax(max_spot, spot);
            }
        } else {
            max_spot = simulate_heston_qe_max_spot(row, num_steps, path, dt);
        }
        const double payoff = discount * fmax(max_spot - row.strike, 0.0);
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

__global__ void heston_lookback_fixed_finalize_kernel(
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
        const double mean = shared[0] / static_cast<double>(num_paths);
        const double variance =
            (shared[blockDim.x] - static_cast<double>(num_paths) * mean * mean)
            / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {
            mean,
            sqrt(fmax(variance, 0.0)) / sqrt(static_cast<double>(num_paths)),
        };
    }
}

__global__ void heston_lookback_fixed_delta_crn_partial_kernel(
    const HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double relative_bump,
    double* partial_price_sums,
    double* partial_price_sumsq,
    double* partial_delta_sums,
    double* partial_delta_sumsq
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
    double local_price_sum = 0.0;
    double local_price_sumsq = 0.0;
    double local_delta_sum = 0.0;
    double local_delta_sumsq = 0.0;

    if (path < num_paths) {
        double spot = row.spot;
        double max_spot = row.spot;
        double variance = row.initial_variance;
        if (row.scheme == kHestonEulerFullTruncation) {
            std::size_t step = 0;
            for (; step + 1U < num_steps;) {
                const auto normal_index = path * num_steps + step;
                if ((normal_index % 2ULL) != 0ULL) {
                    break;
                }
                const auto pair_index = normal_index / 2ULL;
                const auto spot_pair = standard_normal_pair(row.seed, 0, pair_index);
                const auto variance_pair = standard_normal_pair(row.seed, 1, pair_index);
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
                max_spot = fmax(max_spot, spot);
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
                max_spot = fmax(max_spot, spot);
                step += 2U;
            }
            for (; step < num_steps; ++step) {
                const auto normal_index = path * num_steps + step;
                advance_heston_step(
                    row,
                    dt,
                    sqrt_dt,
                    drift_scale,
                    correlation_scale,
                    kappa_dt,
                    vol_of_var_sqrt_dt,
                    standard_normal(row.seed, 0, normal_index),
                    standard_normal(row.seed, 1, normal_index),
                    spot,
                    variance
                );
                max_spot = fmax(max_spot, spot);
            }
        } else {
            max_spot = simulate_heston_qe_max_spot(row, num_steps, path, dt);
        }
        const double payoff = discount * fmax(max_spot - row.strike, 0.0);
        const double up_payoff =
            discount * fmax(max_spot * (1.0 + relative_bump) - row.strike, 0.0);
        const double down_payoff =
            discount * fmax(max_spot * (1.0 - relative_bump) - row.strike, 0.0);
        const double delta =
            (up_payoff - down_payoff) / (2.0 * relative_bump * row.spot);
        local_price_sum = payoff;
        local_price_sumsq = payoff * payoff;
        local_delta_sum = delta;
        local_delta_sumsq = delta * delta;
    }

    reduce_block_four(
        local_price_sum,
        local_price_sumsq,
        local_delta_sum,
        local_delta_sumsq
    );
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto partial_index = row_index * path_blocks_per_row + path_block;
        partial_price_sums[partial_index] = shared[0];
        partial_price_sumsq[partial_index] = shared[blockDim.x];
        partial_delta_sums[partial_index] = shared[2U * blockDim.x];
        partial_delta_sumsq[partial_index] = shared[3U * blockDim.x];
    }
}

__global__ void heston_lookback_fixed_delta_crn_finalize_kernel(
    const double* partial_price_sums,
    const double* partial_price_sumsq,
    const double* partial_delta_sums,
    const double* partial_delta_sumsq,
    std::size_t row_count,
    std::size_t path_blocks_per_row,
    std::size_t num_paths,
    PriceDeltaOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }

    double price_sum = 0.0;
    double price_sumsq = 0.0;
    double delta_sum = 0.0;
    double delta_sumsq = 0.0;
    for (std::size_t block = threadIdx.x; block < path_blocks_per_row; block += blockDim.x) {
        const auto partial_index = row_index * path_blocks_per_row + block;
        price_sum += partial_price_sums[partial_index];
        price_sumsq += partial_price_sumsq[partial_index];
        delta_sum += partial_delta_sums[partial_index];
        delta_sumsq += partial_delta_sumsq[partial_index];
    }

    reduce_block_four(price_sum, price_sumsq, delta_sum, delta_sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const double path_count = static_cast<double>(num_paths);
        const double price_mean = shared[0] / path_count;
        const double price_variance =
            (shared[blockDim.x] - path_count * price_mean * price_mean)
            / static_cast<double>(num_paths - 1U);
        const double delta_mean = shared[2U * blockDim.x] / path_count;
        const double delta_variance =
            (shared[3U * blockDim.x] - path_count * delta_mean * delta_mean)
            / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {
            price_mean,
            sqrt(fmax(price_variance, 0.0)) / sqrt(path_count),
            delta_mean,
            sqrt(fmax(delta_variance, 0.0)) / sqrt(path_count),
        };
    }
}


}  // namespace

struct HestonCudaWorkspace {
    DeviceWorkspace<HestonRow> device;
};

void warmup_cuda() {
    check_cuda(cudaFree(nullptr), "CUDA context warm-up");
}

HestonCudaWorkspace* create_heston_workspace(std::size_t row_capacity) {
    auto* workspace = new HestonCudaWorkspace{};
    try {
        allocate_workspace(workspace->device, row_capacity);
    } catch (...) {
        delete workspace;
        throw;
    }
    return workspace;
}
void destroy_heston_workspace(HestonCudaWorkspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    release_workspace(workspace->device);
    delete workspace;
}
void price_heston_cuda_workspace(
    HestonCudaWorkspace* workspace,
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    run_kernel_with_workspace(
        workspace->device,
        host_rows,
        row_count,
        num_paths,
        num_steps,
        host_outputs,
        timing,
        heston_kernel
    );
}
void price_heston_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    run_kernel_common(
        host_rows,
        row_count,
        num_paths,
        num_steps,
        host_outputs,
        timing,
        heston_kernel
    );
}

void price_heston_lookback_fixed_cuda(
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

    auto& workspace = reusable_cuda_workspace<HestonLookbackWorkspaceTag, 3U>();
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
    check_cuda(
        cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy rows"
    );
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");

    const auto partial_block_count =
        static_cast<unsigned int>(row_count * path_blocks_per_row);
    heston_lookback_fixed_partial_kernel<<<
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
    check_cuda(cudaGetLastError(), "heston lookback partial kernel launch");

    heston_lookback_fixed_finalize_kernel<<<
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
    check_cuda(cudaGetLastError(), "heston lookback finalize kernel launch");
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
}

void price_heston_lookback_fixed_delta_crn_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
) {
    const auto row_bytes = row_count * sizeof(HestonRow);
    const auto output_bytes = row_count * sizeof(PriceDeltaOutput);
    const auto path_blocks_per_row =
        (num_paths + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    const auto partial_count = row_count * path_blocks_per_row;
    const auto shared_bytes = 4U * kThreadsPerBlock * sizeof(double);

    auto& workspace = reusable_cuda_workspace<HestonLookbackWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<HestonRow>(0U, row_count, "cudaMalloc rows");
    auto* device_outputs = workspace.buffer<PriceDeltaOutput>(
        1U, row_count, "cudaMalloc outputs"
    );
    auto* device_partials = workspace.buffer<double>(
        2U, 4U * partial_count, "cudaMalloc partials"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    double* const device_partial_price_sums = device_partials;
    double* const device_partial_price_sumsq = device_partials + partial_count;
    double* const device_partial_delta_sums = device_partials + 2U * partial_count;
    double* const device_partial_delta_sumsq = device_partials + 3U * partial_count;
    check_cuda(
        cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy rows"
    );
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");

    const auto partial_block_count =
        static_cast<unsigned int>(row_count * path_blocks_per_row);
    heston_lookback_fixed_delta_crn_partial_kernel<<<
        partial_block_count,
        kThreadsPerBlock,
        shared_bytes
    >>>(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        relative_bump,
        device_partial_price_sums,
        device_partial_price_sumsq,
        device_partial_delta_sums,
        device_partial_delta_sumsq
    );
    check_cuda(cudaGetLastError(), "heston lookback delta partial kernel launch");

    heston_lookback_fixed_delta_crn_finalize_kernel<<<
        static_cast<unsigned int>(row_count),
        kThreadsPerBlock,
        shared_bytes
    >>>(
        device_partial_price_sums,
        device_partial_price_sumsq,
        device_partial_delta_sums,
        device_partial_delta_sumsq,
        row_count,
        path_blocks_per_row,
        num_paths,
        device_outputs
    );
    check_cuda(cudaGetLastError(), "heston lookback delta finalize kernel launch");
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
}

}  // namespace ai_factory::cuda
