#include "ai_factory/cuda/rough_bergomi/api.cuh"
#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/rough_bergomi/dynamics.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::cuda {
namespace {

using rng::NormalPair;
using rng::RandomQuad;
using rng::standard_normal;
using rng::standard_normal_pair;
using rng::standard_normal_quad;
using rng::standard_uniform;
using rng::standard_uniform_quad;
using reductions::reduce_block;
using reductions::reduce_block_four;
using detail::DeviceWorkspace;
using detail::allocate_workspace;
using detail::check_cuda;
using detail::kThreadsPerBlock;
using detail::reusable_cuda_workspace;
using detail::release_workspace;
using detail::run_kernel_with_workspace;
using rough_bergomi_detail::kMaxRoughBergomiSteps;
using rough_bergomi_detail::rough_bergomi_optimal_weight;
using rough_bergomi_detail::simulate_rough_bergomi_max_spot;

struct RoughBergomiLookbackWorkspaceTag {};

__global__ void rough_bergomi_kernel(
    const RoughBergomiRow* rows,
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
    const double drift = (row.risk_free_rate - row.dividend_yield) * dt;
    const double rho_perp = sqrt(1.0 - row.rho * row.rho);
    const double variance_scale = sqrt(2.0 * row.alpha + 1.0);
    const double singular_covariance_scale = pow(dt, row.alpha + 0.5)
                                             / (row.alpha + 1.0);
    const double singular_residual_variance =
        pow(dt, 2.0 * row.alpha + 1.0)
        * (1.0 / (2.0 * row.alpha + 1.0)
           - 1.0 / ((row.alpha + 1.0) * (row.alpha + 1.0)));
    const double singular_residual_scale =
        sqrt(fmax(singular_residual_variance, 0.0));
    const double discount = exp(-row.risk_free_rate * row.maturity);
    double local_sum = 0.0;
    double local_sumsq = 0.0;
    __shared__ double weights[kMaxRoughBergomiSteps + 1U];
    if (num_steps <= kMaxRoughBergomiSteps) {
        for (std::size_t k = threadIdx.x; k <= num_steps; k += blockDim.x) {
            if (k < 2U) {
                weights[k] = 0.0;
                continue;
            }
            weights[k] = rough_bergomi_optimal_weight(row.alpha, dt, k);
        }
    }
    __syncthreads();

    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        double dws[kMaxRoughBergomiSteps];
        double singular_terms[kMaxRoughBergomiSteps];
        if (num_steps <= kMaxRoughBergomiSteps) {
            for (std::size_t step = 0; step < num_steps; ++step) {
                const auto normal_index = path * num_steps + step;
                const double w_normal =
                    standard_normal(row.seed, 0, normal_index);
                dws[step] = sqrt_dt * w_normal;
                singular_terms[step] =
                    singular_covariance_scale * w_normal
                    + singular_residual_scale
                          * standard_normal(row.seed, 1, normal_index);
            }
        }

        double log_spot = log(row.spot);
        double max_spot = row.spot;
        for (std::size_t step = 0; step < num_steps; ++step) {
            double y = 0.0;
            if (num_steps <= kMaxRoughBergomiSteps) {
                if (step > 0U) {
                    y += singular_terms[step - 1U];
                    for (std::size_t k = 2U; k <= step; ++k) {
                        y += weights[k] * dws[step - k];
                    }
                }
            } else {
                if (step > 0U) {
                    const auto last_index = path * num_steps + step - 1U;
                    const double last_w_normal =
                        standard_normal(row.seed, 0, last_index);
                    const double last_singular_normal =
                        standard_normal(row.seed, 1, last_index);
                    y += singular_covariance_scale * last_w_normal
                         + singular_residual_scale * last_singular_normal;
                    for (std::size_t k = 2U; k <= step; ++k) {
                        const auto past_index = path * num_steps + step - k;
                        const double dw =
                            sqrt_dt * standard_normal(row.seed, 0, past_index);
                        y += rough_bergomi_optimal_weight(row.alpha, dt, k) * dw;
                    }
                }
            }
            const double time = static_cast<double>(step) * dt;
            const double variance =
                row.forward_variance
                * exp(
                    row.eta * variance_scale * y
                    - 0.5 * row.eta * row.eta
                          * pow(time, 2.0 * row.alpha + 1.0)
            );
            const auto normal_index = path * num_steps + step;
            const double dw =
                num_steps <= kMaxRoughBergomiSteps
                    ? dws[step]
                    : sqrt_dt * standard_normal(row.seed, 0, normal_index);
            const double dz =
                row.rho * dw
                + rho_perp * sqrt_dt
                      * standard_normal(row.seed, 2, normal_index);
            log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
            max_spot = fmax(max_spot, exp(log_spot));
        }
        const double payoff = fmax(max_spot - row.strike, 0.0);
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

[[maybe_unused]] __device__ void rough_bergomi_delta_crn_kernel(
    const RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto row = rows[row_index];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift = (row.risk_free_rate - row.dividend_yield) * dt;
    const double rho_perp = sqrt(1.0 - row.rho * row.rho);
    const double variance_scale = sqrt(2.0 * row.alpha + 1.0);
    const double singular_covariance_scale = pow(dt, row.alpha + 0.5)
                                             / (row.alpha + 1.0);
    const double singular_residual_variance =
        pow(dt, 2.0 * row.alpha + 1.0)
        * (1.0 / (2.0 * row.alpha + 1.0)
           - 1.0 / ((row.alpha + 1.0) * (row.alpha + 1.0)));
    const double singular_residual_scale =
        sqrt(fmax(singular_residual_variance, 0.0));
    const double discount = exp(-row.risk_free_rate * row.maturity);
    double local_price_sum = 0.0;
    double local_price_sumsq = 0.0;
    double local_delta_sum = 0.0;
    double local_delta_sumsq = 0.0;
    __shared__ double weights[kMaxRoughBergomiSteps + 1U];
    if (num_steps <= kMaxRoughBergomiSteps) {
        for (std::size_t k = threadIdx.x; k <= num_steps; k += blockDim.x) {
            if (k < 2U) {
                weights[k] = 0.0;
                continue;
            }
            weights[k] = rough_bergomi_optimal_weight(row.alpha, dt, k);
        }
    }
    __syncthreads();

    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        double dws[kMaxRoughBergomiSteps];
        double singular_terms[kMaxRoughBergomiSteps];
        if (num_steps <= kMaxRoughBergomiSteps) {
            for (std::size_t step = 0; step < num_steps; ++step) {
                const auto normal_index = path * num_steps + step;
                const double w_normal =
                    standard_normal(row.seed, 0, normal_index);
                dws[step] = sqrt_dt * w_normal;
                singular_terms[step] =
                    singular_covariance_scale * w_normal
                    + singular_residual_scale
                          * standard_normal(row.seed, 1, normal_index);
            }
        }

        double log_spot = log(row.spot);
        double max_spot = row.spot;
        for (std::size_t step = 0; step < num_steps; ++step) {
            double y = 0.0;
            if (num_steps <= kMaxRoughBergomiSteps) {
                if (step > 0U) {
                    y += singular_terms[step - 1U];
                    for (std::size_t k = 2U; k <= step; ++k) {
                        y += weights[k] * dws[step - k];
                    }
                }
            } else {
                if (step > 0U) {
                    const auto last_index = path * num_steps + step - 1U;
                    const double last_w_normal =
                        standard_normal(row.seed, 0, last_index);
                    const double last_singular_normal =
                        standard_normal(row.seed, 1, last_index);
                    y += singular_covariance_scale * last_w_normal
                         + singular_residual_scale * last_singular_normal;
                    for (std::size_t k = 2U; k <= step; ++k) {
                        const auto past_index = path * num_steps + step - k;
                        const double dw =
                            sqrt_dt * standard_normal(row.seed, 0, past_index);
                        y += rough_bergomi_optimal_weight(row.alpha, dt, k) * dw;
                    }
                }
            }
            const double time = static_cast<double>(step) * dt;
            const double variance =
                row.forward_variance
                * exp(
                    row.eta * variance_scale * y
                    - 0.5 * row.eta * row.eta
                          * pow(time, 2.0 * row.alpha + 1.0)
            );
            const auto normal_index = path * num_steps + step;
            const double dw =
                num_steps <= kMaxRoughBergomiSteps
                    ? dws[step]
                    : sqrt_dt * standard_normal(row.seed, 0, normal_index);
            const double dz =
                row.rho * dw
                + rho_perp * sqrt_dt
                      * standard_normal(row.seed, 2, normal_index);
            log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
            max_spot = fmax(max_spot, exp(log_spot));
        }
        const double price_payoff = discount * fmax(max_spot - row.strike, 0.0);
        const double up_payoff =
            discount * fmax(max_spot * (1.0 + relative_bump) - row.strike, 0.0);
        const double down_payoff =
            discount * fmax(max_spot * (1.0 - relative_bump) - row.strike, 0.0);
        const double delta =
            (up_payoff - down_payoff) / (2.0 * relative_bump * row.spot);
        local_price_sum += price_payoff;
        local_price_sumsq += price_payoff * price_payoff;
        local_delta_sum += delta;
        local_delta_sumsq += delta * delta;
    }

    reduce_block_four(
        local_price_sum,
        local_price_sumsq,
        local_delta_sum,
        local_delta_sumsq
    );
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

template <std::size_t MaxSteps, bool ComputeDelta>
__global__ void rough_bergomi_lookback_partial_kernel(
    const RoughBergomiRow* rows,
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
    if constexpr (!ComputeDelta) {
        (void)relative_bump;
        (void)partial_delta_sums;
        (void)partial_delta_sumsq;
    }
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
    __shared__ double weights[MaxSteps + 1U];
    __shared__ double variance_time_powers[MaxSteps];
    for (std::size_t step = threadIdx.x; step < num_steps; step += blockDim.x) {
        if (step >= 2U) {
            weights[step] = rough_bergomi_optimal_weight(row.alpha, dt, step);
        } else {
            weights[step] = 0.0;
        }
        const double time = static_cast<double>(step) * dt;
        variance_time_powers[step] = pow(time, 2.0 * row.alpha + 1.0);
    }
    __syncthreads();

    double price = 0.0;
    [[maybe_unused]] double delta = 0.0;
    if (path < num_paths) {
        const double max_spot = simulate_rough_bergomi_max_spot<MaxSteps>(
            row, num_steps, path, weights, variance_time_powers
        );
        const double discount = exp(-row.risk_free_rate * row.maturity);
        price = discount * fmax(max_spot - row.strike, 0.0);
        if constexpr (ComputeDelta) {
            const double up_payoff =
                discount
                * fmax(max_spot * (1.0 + relative_bump) - row.strike, 0.0);
            const double down_payoff =
                discount
                * fmax(max_spot * (1.0 - relative_bump) - row.strike, 0.0);
            delta =
                (up_payoff - down_payoff)
                / (2.0 * relative_bump * row.spot);
        }
    }

    if constexpr (ComputeDelta) {
        reduce_block_four(price, price * price, delta, delta * delta);
    } else {
        reduce_block(price, price * price);
    }
    if (threadIdx.x == 0U) {
        extern __shared__ double shared[];
        const auto partial_index = row_index * path_blocks_per_row + path_block;
        partial_price_sums[partial_index] = shared[0];
        partial_price_sumsq[partial_index] = shared[blockDim.x];
        if constexpr (ComputeDelta) {
            partial_delta_sums[partial_index] = shared[2U * blockDim.x];
            partial_delta_sumsq[partial_index] = shared[3U * blockDim.x];
        }
    }
}

__global__ void lookback_fixed_finalize_kernel(
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

__global__ void lookback_fixed_delta_crn_finalize_kernel(
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

template <std::size_t MaxSteps, bool ComputeDelta>
void launch_rough_bergomi_partial_kernel(
    const RoughBergomiRow* rows,
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
    const auto block_count =
        static_cast<unsigned int>(row_count * path_blocks_per_row);
    const auto shared_bytes =
        (ComputeDelta ? 4U : 2U) * kThreadsPerBlock * sizeof(double);
    rough_bergomi_lookback_partial_kernel<MaxSteps, ComputeDelta><<<
        block_count,
        kThreadsPerBlock,
        shared_bytes
    >>>(
        rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        relative_bump,
        partial_price_sums,
        partial_price_sumsq,
        partial_delta_sums,
        partial_delta_sumsq
    );
}

template <bool ComputeDelta>
void dispatch_rough_bergomi_partial_kernel(
    const RoughBergomiRow* rows,
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
    if (num_steps <= 16U) {
        launch_rough_bergomi_partial_kernel<16U, ComputeDelta>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row,
            relative_bump, partial_price_sums, partial_price_sumsq,
            partial_delta_sums, partial_delta_sumsq
        );
    } else if (num_steps <= 32U) {
        launch_rough_bergomi_partial_kernel<32U, ComputeDelta>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row,
            relative_bump, partial_price_sums, partial_price_sumsq,
            partial_delta_sums, partial_delta_sumsq
        );
    } else if (num_steps <= 64U) {
        launch_rough_bergomi_partial_kernel<64U, ComputeDelta>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row,
            relative_bump, partial_price_sums, partial_price_sumsq,
            partial_delta_sums, partial_delta_sumsq
        );
    } else if (num_steps <= 128U) {
        launch_rough_bergomi_partial_kernel<128U, ComputeDelta>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row,
            relative_bump, partial_price_sums, partial_price_sumsq,
            partial_delta_sums, partial_delta_sumsq
        );
    } else if (num_steps <= 160U) {
        launch_rough_bergomi_partial_kernel<160U, ComputeDelta>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row,
            relative_bump, partial_price_sums, partial_price_sumsq,
            partial_delta_sums, partial_delta_sumsq
        );
    } else if (num_steps <= kMaxRoughBergomiSteps) {
        launch_rough_bergomi_partial_kernel<kMaxRoughBergomiSteps, ComputeDelta>(
            rows, row_count, num_paths, num_steps, path_blocks_per_row,
            relative_bump, partial_price_sums, partial_price_sumsq,
            partial_delta_sums, partial_delta_sumsq
        );
    } else {
        throw std::invalid_argument(
            "Optimized rough Bergomi CUDA kernels support at most 256 steps."
        );
    }
    check_cuda(cudaGetLastError(), "rough Bergomi partial kernel launch");
}

}  // namespace


struct RoughBergomiCudaWorkspace {
    DeviceWorkspace<RoughBergomiRow> device;
};
RoughBergomiCudaWorkspace* create_rough_bergomi_workspace(
    std::size_t row_capacity
) {
    auto* workspace = new RoughBergomiCudaWorkspace{};
    try {
        allocate_workspace(workspace->device, row_capacity);
    } catch (...) {
        delete workspace;
        throw;
    }
    return workspace;
}
void destroy_rough_bergomi_workspace(RoughBergomiCudaWorkspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    release_workspace(workspace->device);
    delete workspace;
}
void price_rough_bergomi_cuda_workspace(
    RoughBergomiCudaWorkspace* workspace,
    const RoughBergomiRow* host_rows,
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
        rough_bergomi_kernel
    );
}
void price_rough_bergomi_cuda(
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    const auto row_bytes = row_count * sizeof(RoughBergomiRow);
    const auto output_bytes = row_count * sizeof(MonteCarloOutput);
    const auto path_blocks_per_row =
        (num_paths + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    const auto partial_count = row_count * path_blocks_per_row;
    const auto shared_bytes = 2U * kThreadsPerBlock * sizeof(double);

    auto& workspace = reusable_cuda_workspace<RoughBergomiLookbackWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<RoughBergomiRow>(0U, row_count, "cudaMalloc rough rows");
    auto* device_outputs = workspace.buffer<MonteCarloOutput>(1U, row_count, "cudaMalloc rough outputs");
    auto* device_partials = workspace.buffer<double>(2U, 2U * partial_count, "cudaMalloc rough partials");
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    double* const partial_sums = device_partials;
    double* const partial_sumsq = device_partials + partial_count;
    try {
        check_cuda(
            cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy rough rows"
        );
        check_cuda(cudaEventRecord(start), "cudaEventRecord rough start");
        dispatch_rough_bergomi_partial_kernel<false>(
            device_rows,
            row_count,
            num_paths,
            num_steps,
            path_blocks_per_row,
            0.0,
            partial_sums,
            partial_sumsq,
            nullptr,
            nullptr
        );
        lookback_fixed_finalize_kernel<<<
            static_cast<unsigned int>(row_count),
            kThreadsPerBlock,
            shared_bytes
        >>>(
            partial_sums,
            partial_sumsq,
            row_count,
            path_blocks_per_row,
            num_paths,
            device_outputs
        );
        check_cuda(cudaGetLastError(), "rough Bergomi finalize kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord rough stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize rough stop");
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "rough timing");
        check_cuda(
            cudaMemcpy(host_outputs, device_outputs, output_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy rough outputs"
        );
        if (timing != nullptr) {
            timing->simulation_ms = elapsed_ms;
            timing->total_ms = elapsed_ms;
        }
    } catch (...) {
        throw;
    }
}

void price_rough_bergomi_delta_crn_cuda(
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
) {
    const auto row_bytes = row_count * sizeof(RoughBergomiRow);
    const auto output_bytes = row_count * sizeof(PriceDeltaOutput);
    const auto path_blocks_per_row =
        (num_paths + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    const auto partial_count = row_count * path_blocks_per_row;
    const auto shared_bytes = 4U * kThreadsPerBlock * sizeof(double);

    auto& workspace = reusable_cuda_workspace<RoughBergomiLookbackWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<RoughBergomiRow>(0U, row_count, "cudaMalloc rough rows");
    auto* device_outputs = workspace.buffer<PriceDeltaOutput>(1U, row_count, "cudaMalloc rough delta outputs");
    auto* device_partials = workspace.buffer<double>(2U, 4U * partial_count, "cudaMalloc rough delta partials");
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    double* const partial_price_sums = device_partials;
    double* const partial_price_sumsq = device_partials + partial_count;
    double* const partial_delta_sums = device_partials + 2U * partial_count;
    double* const partial_delta_sumsq = device_partials + 3U * partial_count;
    try {
        check_cuda(
            cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy rough rows"
        );
        check_cuda(cudaEventRecord(start), "cudaEventRecord rough delta start");

        dispatch_rough_bergomi_partial_kernel<true>(
            device_rows,
            row_count,
            num_paths,
            num_steps,
            path_blocks_per_row,
            relative_bump,
            partial_price_sums,
            partial_price_sumsq,
            partial_delta_sums,
            partial_delta_sumsq
        );
        lookback_fixed_delta_crn_finalize_kernel<<<
            static_cast<unsigned int>(row_count),
            kThreadsPerBlock,
            shared_bytes
        >>>(
            partial_price_sums,
            partial_price_sumsq,
            partial_delta_sums,
            partial_delta_sumsq,
            row_count,
            path_blocks_per_row,
            num_paths,
            device_outputs
        );
        check_cuda(cudaGetLastError(), "rough Bergomi delta finalize kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord rough delta stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize rough delta stop");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "rough delta timing");
        check_cuda(
            cudaMemcpy(host_outputs, device_outputs, output_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy rough delta outputs"
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
