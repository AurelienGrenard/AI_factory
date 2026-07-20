#include "ai_factory/cuda/black_scholes/api.cuh"
#include "ai_factory/cuda/black_scholes/dynamics.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::cuda {
namespace {

using reductions::reduce_block;
using detail::check_cuda;
using detail::kThreadsPerBlock;
using detail::reusable_cuda_workspace;
using black_scholes_detail::simulate_realized_volatility;

struct BlackScholesVolatilityWorkspaceTag {};

__global__ void volatility_swap_kernel(
    const BlackScholesRow* rows,
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
    const double discount = exp(-row.risk_free_rate * row.maturity);
    double local_sum = 0.0;
    double local_sumsq = 0.0;
    const auto path =
        path_block * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (path < num_paths) {
        const double realized_vol = simulate_realized_volatility(row, path, num_steps);
        const double payoff = discount * (realized_vol - row.strike);
        local_sum += payoff;
        local_sumsq += payoff * payoff;
    }
    reduce_block(local_sum, local_sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto partial_index = row_index * path_blocks_per_row + path_block;
        partial_sums[partial_index] = shared[0];
        partial_sumsq[partial_index] = shared[blockDim.x];
    }
}

__global__ void volatility_swap_finalize_kernel(
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
            (shared[blockDim.x] - path_count * mean * mean) / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {mean, sqrt(fmax(variance, 0.0)) / sqrt(path_count)};
    }
}

}  // namespace

void price_black_scholes_volatility_swap_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    const auto row_bytes = row_count * sizeof(BlackScholesRow);
    const auto output_bytes = row_count * sizeof(MonteCarloOutput);
    const auto path_blocks_per_row =
        (num_paths + static_cast<std::size_t>(kThreadsPerBlock) - 1U)
        / static_cast<std::size_t>(kThreadsPerBlock);
    const auto partial_count = row_count * path_blocks_per_row;
    auto& workspace = reusable_cuda_workspace<BlackScholesVolatilityWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<BlackScholesRow>(0U, row_count, "cudaMalloc rows");
    auto* device_outputs = workspace.buffer<MonteCarloOutput>(1U, row_count, "cudaMalloc outputs");
    auto* device_partials = workspace.buffer<double>(2U, 2U * partial_count, "cudaMalloc partials");
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    check_cuda(cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice), "cudaMemcpy rows");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    volatility_swap_kernel<<<
        static_cast<unsigned int>(partial_count),
        kThreadsPerBlock,
        2U * kThreadsPerBlock * sizeof(double)
    >>>(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        device_partials,
        device_partials + partial_count
    );
    check_cuda(cudaGetLastError(), "black_scholes volatility swap kernel");
    volatility_swap_finalize_kernel<<<
        static_cast<unsigned int>(row_count),
        kThreadsPerBlock,
        2U * kThreadsPerBlock * sizeof(double)
    >>>(
        device_partials,
        device_partials + partial_count,
        row_count,
        path_blocks_per_row,
        num_paths,
        device_outputs
    );
    check_cuda(cudaGetLastError(), "black_scholes volatility swap finalize kernel");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing");
    check_cuda(cudaMemcpy(host_outputs, device_outputs, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy outputs");
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
}

}  // namespace ai_factory::cuda
