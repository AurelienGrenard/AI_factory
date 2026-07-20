#include "ai_factory/cuda/rough_bergomi/api.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/rough_bergomi/dynamics.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::cuda {
namespace {

using reductions::reduce_block;
using reductions::reduce_block_four;
using detail::check_cuda;
using detail::kThreadsPerBlock;
using detail::reusable_cuda_workspace;
using rough_bergomi_detail::kMaxRoughBergomiSteps;
using rough_bergomi_detail::rough_bergomi_optimal_weight;
using rough_bergomi_detail::simulate_rough_bergomi_average_spot;

struct RoughBergomiAsianWorkspaceTag {};

template <std::size_t MaxSteps, bool ComputeDelta>
__global__ void rough_bergomi_asian_arithmetic_partial_kernel(
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
        const double average_spot = simulate_rough_bergomi_average_spot<MaxSteps>(
            row, num_steps, path, weights, variance_time_powers
        );
        const double discount = exp(-row.risk_free_rate * row.maturity);
        price = discount * fmax(average_spot - row.strike, 0.0);
        if constexpr (ComputeDelta) {
            const double up_payoff =
                discount
                * fmax(average_spot * (1.0 + relative_bump) - row.strike, 0.0);
            const double down_payoff =
                discount
                * fmax(average_spot * (1.0 - relative_bump) - row.strike, 0.0);
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

__global__ void asian_arithmetic_finalize_kernel(
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

__global__ void asian_arithmetic_delta_crn_finalize_kernel(
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
    rough_bergomi_asian_arithmetic_partial_kernel<MaxSteps, ComputeDelta><<<
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


void price_rough_bergomi_asian_arithmetic_cuda(
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

    auto& workspace = reusable_cuda_workspace<RoughBergomiAsianWorkspaceTag, 3U>();
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
        asian_arithmetic_finalize_kernel<<<
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

void price_rough_bergomi_asian_arithmetic_delta_crn_cuda(
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

    auto& workspace = reusable_cuda_workspace<RoughBergomiAsianWorkspaceTag, 3U>();
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
        asian_arithmetic_delta_crn_finalize_kernel<<<
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
