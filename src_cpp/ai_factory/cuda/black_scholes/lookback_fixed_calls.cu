#include "ai_factory/cuda/black_scholes/api.cuh"
#include "ai_factory/cuda/black_scholes/dynamics.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::cuda {
namespace {

using reductions::reduce_block;
using reductions::reduce_block_four;
using detail::check_cuda;
using detail::kThreadsPerBlock;
using detail::reusable_cuda_workspace;
using black_scholes_detail::simulate_max_spot;

struct BlackScholesLookbackWorkspaceTag {};

__global__ void lookback_kernel(
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
        const double max_spot = simulate_max_spot(row, path, num_steps);
        const double payoff = discount * fmax(max_spot - row.strike, 0.0);
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

__global__ void lookback_delta_kernel(
    const BlackScholesRow* rows,
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
    const double discount = exp(-row.risk_free_rate * row.maturity);
    double price_sum = 0.0;
    double price_sumsq = 0.0;
    double delta_sum = 0.0;
    double delta_sumsq = 0.0;
    const auto path =
        path_block * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (path < num_paths) {
        const double max_spot = simulate_max_spot(row, path, num_steps);
        const double price = discount * fmax(max_spot - row.strike, 0.0);
        const double up = discount * fmax(max_spot * (1.0 + relative_bump) - row.strike, 0.0);
        const double down = discount * fmax(max_spot * (1.0 - relative_bump) - row.strike, 0.0);
        const double delta = (up - down) / (2.0 * relative_bump * row.spot);
        price_sum += price;
        price_sumsq += price * price;
        delta_sum += delta;
        delta_sumsq += delta * delta;
    }
    reduce_block_four(price_sum, price_sumsq, delta_sum, delta_sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const auto partial_index = row_index * path_blocks_per_row + path_block;
        partial_price_sums[partial_index] = shared[0];
        partial_price_sumsq[partial_index] = shared[blockDim.x];
        partial_delta_sums[partial_index] = shared[2U * blockDim.x];
        partial_delta_sumsq[partial_index] = shared[3U * blockDim.x];
    }
}

__global__ void lookback_finalize_kernel(
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

__global__ void lookback_delta_finalize_kernel(
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
        const double price = shared[0] / path_count;
        const double price_variance =
            (shared[blockDim.x] - path_count * price * price) / static_cast<double>(num_paths - 1U);
        const double delta = shared[2U * blockDim.x] / path_count;
        const double delta_variance =
            (shared[3U * blockDim.x] - path_count * delta * delta)
            / static_cast<double>(num_paths - 1U);
        outputs[row_index] = {
            price,
            sqrt(fmax(price_variance, 0.0)) / sqrt(path_count),
            delta,
            sqrt(fmax(delta_variance, 0.0)) / sqrt(path_count),
        };
    }
}

void run_price_kernel(
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
    auto& workspace = reusable_cuda_workspace<BlackScholesLookbackWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<BlackScholesRow>(0U, row_count, "cudaMalloc rows");
    auto* device_outputs = workspace.buffer<MonteCarloOutput>(1U, row_count, "cudaMalloc outputs");
    auto* device_partials = workspace.buffer<double>(2U, 2U * partial_count, "cudaMalloc partials");
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    check_cuda(cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice), "cudaMemcpy rows");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    lookback_kernel<<<
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
    check_cuda(cudaGetLastError(), "black_scholes lookback kernel");
    lookback_finalize_kernel<<<
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
    check_cuda(cudaGetLastError(), "black_scholes lookback finalize kernel");
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

void run_delta_kernel(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
) {
    const auto row_bytes = row_count * sizeof(BlackScholesRow);
    const auto output_bytes = row_count * sizeof(PriceDeltaOutput);
    const auto path_blocks_per_row =
        (num_paths + static_cast<std::size_t>(kThreadsPerBlock) - 1U)
        / static_cast<std::size_t>(kThreadsPerBlock);
    const auto partial_count = row_count * path_blocks_per_row;
    auto& workspace = reusable_cuda_workspace<BlackScholesLookbackWorkspaceTag, 3U>();
    auto* device_rows = workspace.buffer<BlackScholesRow>(0U, row_count, "cudaMalloc rows");
    auto* device_outputs = workspace.buffer<PriceDeltaOutput>(1U, row_count, "cudaMalloc outputs");
    auto* device_partials = workspace.buffer<double>(2U, 4U * partial_count, "cudaMalloc partials");
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    check_cuda(cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice), "cudaMemcpy rows");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    lookback_delta_kernel<<<
        static_cast<unsigned int>(partial_count),
        kThreadsPerBlock,
        4U * kThreadsPerBlock * sizeof(double)
    >>>(
        device_rows,
        row_count,
        num_paths,
        num_steps,
        path_blocks_per_row,
        relative_bump,
        device_partials,
        device_partials + partial_count,
        device_partials + 2U * partial_count,
        device_partials + 3U * partial_count
    );
    check_cuda(cudaGetLastError(), "black_scholes lookback delta kernel");
    lookback_delta_finalize_kernel<<<
        static_cast<unsigned int>(row_count),
        kThreadsPerBlock,
        4U * kThreadsPerBlock * sizeof(double)
    >>>(
        device_partials,
        device_partials + partial_count,
        device_partials + 2U * partial_count,
        device_partials + 3U * partial_count,
        row_count,
        path_blocks_per_row,
        num_paths,
        device_outputs
    );
    check_cuda(cudaGetLastError(), "black_scholes lookback delta finalize kernel");
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

}  // namespace

void price_black_scholes_lookback_fixed_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    run_price_kernel(host_rows, row_count, num_paths, num_steps, host_outputs, timing);
}

void price_black_scholes_lookback_fixed_delta_crn_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
) {
    run_delta_kernel(host_rows, row_count, num_paths, num_steps, relative_bump, host_outputs, timing);
}

}  // namespace ai_factory::cuda
