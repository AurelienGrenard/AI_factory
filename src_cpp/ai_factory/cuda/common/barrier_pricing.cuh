#pragma once

#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/common/types.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::cuda::barrier_detail {

struct PathState {
    double terminal_spot;
    bool hit;
};

template <typename Row, typename Simulator, bool KnockIn>
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
    double payoff = 0.0;
    if (path < num_paths) {
        const auto state = Simulator::run(
            row.model, path, num_steps, row.product.barrier
        );
        const bool active = KnockIn ? state.hit : !state.hit;
        if (active) {
            payoff = exp(-row.model.risk_free_rate * row.model.maturity)
                     * fmax(state.terminal_spot - row.model.strike, 0.0);
        }
    }
    double sum = payoff;
    double sumsq = payoff * payoff;
    reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto index = row_index * path_blocks_per_row + path_block;
        partial_sums[index] = shared[0];
        partial_sumsq[index] = shared[blockDim.x];
    }
}

template <typename WorkspaceTag>
__global__ void finalize_kernel(
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
        const double variance =
            (shared[blockDim.x] - count * mean * mean)
            / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {
            mean,
            sqrt(fmax(variance, 0.0)) / sqrt(count),
        };
    }
}

template <typename Row, typename Simulator, bool KnockIn, typename WorkspaceTag>
void run(
    const Row* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    if (row_count == 0U || num_paths < 2U || num_steps == 0U) {
        throw std::invalid_argument("Barrier pricing requires rows, steps, and at least two paths.");
    }
    constexpr auto threads = detail::kThreadsPerBlock;
    const auto blocks_per_row =
        (num_paths + static_cast<std::size_t>(threads) - 1U)
        / static_cast<std::size_t>(threads);
    const auto partial_count = row_count * blocks_per_row;
    auto& workspace = detail::reusable_cuda_workspace<WorkspaceTag, 3U>();
    auto* device_rows = workspace.template buffer<Row>(0U, row_count, "cudaMalloc barrier rows");
    auto* device_outputs = workspace.template buffer<MonteCarloOutput>(1U, row_count, "cudaMalloc barrier outputs");
    auto* partials = workspace.template buffer<double>(2U, 2U * partial_count, "cudaMalloc barrier partials");
    detail::check_cuda(
        cudaMemcpy(device_rows, host_rows, row_count * sizeof(Row), cudaMemcpyHostToDevice),
        "cudaMemcpy barrier rows"
    );
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start), "cudaEventRecord barrier start");
    partial_kernel<Row, Simulator, KnockIn><<<
        static_cast<unsigned int>(partial_count),
        threads,
        2U * threads * sizeof(double)
    >>>(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        blocks_per_row,
        partials,
        partials + partial_count
    );
    detail::check_cuda(cudaGetLastError(), "barrier partial kernel");
    finalize_kernel<WorkspaceTag><<<
        static_cast<unsigned int>(row_count),
        threads,
        2U * threads * sizeof(double)
    >>>(
        partials,
        partials + partial_count,
        row_count,
        blocks_per_row,
        num_paths,
        device_outputs
    );
    detail::check_cuda(cudaGetLastError(), "barrier finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "cudaEventRecord barrier stop");
    detail::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize barrier");
    float elapsed_ms = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "barrier timing");
    detail::check_cuda(
        cudaMemcpy(host_outputs, device_outputs, row_count * sizeof(MonteCarloOutput), cudaMemcpyDeviceToHost),
        "cudaMemcpy barrier outputs"
    );
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
}

}  // namespace ai_factory::cuda::barrier_detail
