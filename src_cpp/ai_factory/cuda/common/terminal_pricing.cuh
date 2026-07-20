#pragma once

#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::cuda::terminal_pricing {

enum class Payoff { Call, DigitalCall };

template <Payoff payoff>
__device__ __forceinline__ double evaluate(double terminal, double strike) {
    if constexpr (payoff == Payoff::Call) {
        return fmax(terminal - strike, 0.0);
    }
    return terminal > strike ? 1.0 : 0.0;
}

template <typename Row, typename Simulator, Payoff payoff>
__global__ void partial_kernel(
    const Row* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t path_blocks_per_row,
    double* partial_sums,
    double* partial_sumsq
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x) / path_blocks_per_row;
    const auto path_block = static_cast<std::size_t>(blockIdx.x) % path_blocks_per_row;
    if (row_index >= row_count) {
        return;
    }
    const auto row = rows[row_index];
    const auto path = path_block * static_cast<std::size_t>(blockDim.x)
                      + static_cast<std::size_t>(threadIdx.x);
    double value = 0.0;
    double square = 0.0;
    if (path < num_paths) {
        const double terminal = Simulator{}(row, path, num_steps);
        value = exp(-row.risk_free_rate * row.maturity)
                * evaluate<payoff>(terminal, row.strike);
        square = value * value;
    }
    reductions::reduce_block(value, square);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto index = row_index * path_blocks_per_row + path_block;
        partial_sums[index] = shared[0];
        partial_sumsq[index] = shared[blockDim.x];
    }
}

__global__ inline void finalize_kernel(
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
    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t block = threadIdx.x; block < path_blocks_per_row; block += blockDim.x) {
        const auto index = row_index * path_blocks_per_row + block;
        sum += partial_sums[index];
        sumsq += partial_sumsq[index];
    }
    reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const double count = static_cast<double>(num_paths);
        const double mean = shared[0] / count;
        const double variance = (shared[blockDim.x] - count * mean * mean)
                                / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {mean, sqrt(fmax(variance, 0.0) / count)};
    }
}

template <typename Tag, typename Row, typename Simulator, Payoff payoff>
void run(
    const Row* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    constexpr auto threads = detail::kThreadsPerBlock;
    const auto path_blocks = (num_paths + threads - 1U) / threads;
    const auto partial_count = row_count * path_blocks;
    auto& workspace = detail::reusable_cuda_workspace<Tag, 3U>();
    auto* rows = workspace.template buffer<Row>(0U, row_count, "cudaMalloc terminal rows");
    auto* outputs = workspace.template buffer<MonteCarloOutput>(1U, row_count, "cudaMalloc terminal outputs");
    auto* partials = workspace.template buffer<double>(2U, 2U * partial_count, "cudaMalloc terminal partials");
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaMemcpy(rows, host_rows, row_count * sizeof(Row), cudaMemcpyHostToDevice), "cudaMemcpy terminal rows");
    detail::check_cuda(cudaEventRecord(start), "cudaEventRecord terminal start");
    partial_kernel<Row, Simulator, payoff><<<
        static_cast<unsigned int>(partial_count), threads, 2U * threads * sizeof(double)
    >>>(rows, row_count, num_paths, num_steps, path_blocks, partials, partials + partial_count);
    detail::check_cuda(cudaGetLastError(), "terminal partial kernel");
    finalize_kernel<<<
        static_cast<unsigned int>(row_count), threads, 2U * threads * sizeof(double)
    >>>(partials, partials + partial_count, row_count, path_blocks, num_paths, outputs);
    detail::check_cuda(cudaGetLastError(), "terminal finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "cudaEventRecord terminal stop");
    detail::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize terminal stop");
    float elapsed_ms = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "terminal kernel timing");
    detail::check_cuda(cudaMemcpy(host_outputs, outputs, row_count * sizeof(MonteCarloOutput), cudaMemcpyDeviceToHost), "cudaMemcpy terminal outputs");
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
}

}  // namespace ai_factory::cuda::terminal_pricing
