#pragma once

#include "ai_factory/cuda/common/terminal_pricing.cuh"
#include "ai_factory/cuda/rough_bergomi/dynamics.cuh"

#include <stdexcept>

namespace ai_factory::cuda::rough_bergomi_terminal {

template <terminal_pricing::Payoff payoff>
__global__ void partial_kernel(
    const RoughBergomiRow* rows, std::size_t row_count, std::size_t num_paths,
    std::size_t num_steps, std::size_t path_blocks_per_row,
    double* partial_sums, double* partial_sumsq
) {
    constexpr auto max_steps = rough_bergomi_detail::kMaxRoughBergomiSteps;
    const auto row_index = static_cast<std::size_t>(blockIdx.x) / path_blocks_per_row;
    const auto path_block = static_cast<std::size_t>(blockIdx.x) % path_blocks_per_row;
    if (row_index >= row_count) return;
    const auto row = rows[row_index];
    extern __shared__ double shared[];
    double* weights = shared + 2U * blockDim.x;
    double* powers = weights + max_steps + 1U;
    const double dt = row.maturity / static_cast<double>(num_steps);
    for (std::size_t step = threadIdx.x; step < num_steps; step += blockDim.x) {
        powers[step] = pow(static_cast<double>(step) * dt, 2.0 * row.alpha + 1.0);
    }
    for (std::size_t k = threadIdx.x; k <= num_steps; k += blockDim.x) {
        weights[k] = k < 2U ? 0.0
            : rough_bergomi_detail::rough_bergomi_optimal_weight(row.alpha, dt, k);
    }
    __syncthreads();
    const auto path = path_block * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    double value = 0.0;
    double square = 0.0;
    if (path < num_paths) {
        const double terminal = rough_bergomi_detail::simulate_rough_bergomi_terminal_spot<max_steps>(
            row, num_steps, path, weights, powers
        );
        value = exp(-row.risk_free_rate * row.maturity)
                * terminal_pricing::evaluate<payoff>(terminal, row.strike);
        square = value * value;
    }
    reductions::reduce_block(value, square);
    if (threadIdx.x == 0) {
        const auto index = row_index * path_blocks_per_row + path_block;
        partial_sums[index] = shared[0];
        partial_sumsq[index] = shared[blockDim.x];
    }
}

template <typename Tag, terminal_pricing::Payoff payoff>
void run(
    const RoughBergomiRow* host_rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    MonteCarloOutput* host_outputs, CudaTiming* timing
) {
    if (num_steps > rough_bergomi_detail::kMaxRoughBergomiSteps) {
        throw std::invalid_argument("Rough Bergomi terminal kernel step limit exceeded.");
    }
    constexpr auto threads = detail::kThreadsPerBlock;
    const auto path_blocks = (num_paths + threads - 1U) / threads;
    const auto partial_count = row_count * path_blocks;
    auto& workspace = detail::reusable_cuda_workspace<Tag, 3U>();
    auto* rows = workspace.template buffer<RoughBergomiRow>(0U, row_count, "cudaMalloc rough terminal rows");
    auto* outputs = workspace.template buffer<MonteCarloOutput>(1U, row_count, "cudaMalloc rough terminal outputs");
    auto* partials = workspace.template buffer<double>(2U, 2U * partial_count, "cudaMalloc rough terminal partials");
    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaMemcpy(rows, host_rows, row_count * sizeof(RoughBergomiRow), cudaMemcpyHostToDevice), "cudaMemcpy rough terminal rows");
    detail::check_cuda(cudaEventRecord(start), "cudaEventRecord rough terminal start");
    const auto shared_bytes = (2U * threads + 2U * (rough_bergomi_detail::kMaxRoughBergomiSteps + 1U)) * sizeof(double);
    partial_kernel<payoff><<<static_cast<unsigned int>(partial_count), threads, shared_bytes>>>(
        rows, row_count, num_paths, num_steps, path_blocks, partials, partials + partial_count
    );
    detail::check_cuda(cudaGetLastError(), "rough terminal partial kernel");
    terminal_pricing::finalize_kernel<<<static_cast<unsigned int>(row_count), threads, 2U * threads * sizeof(double)>>>(
        partials, partials + partial_count, row_count, path_blocks, num_paths, outputs
    );
    detail::check_cuda(cudaGetLastError(), "rough terminal finalize kernel");
    detail::check_cuda(cudaEventRecord(stop), "cudaEventRecord rough terminal stop");
    detail::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize rough terminal stop");
    float elapsed_ms = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "rough terminal timing");
    detail::check_cuda(cudaMemcpy(host_outputs, outputs, row_count * sizeof(MonteCarloOutput), cudaMemcpyDeviceToHost), "cudaMemcpy rough terminal outputs");
    if (timing != nullptr) { timing->simulation_ms = elapsed_ms; timing->total_ms = elapsed_ms; }
}

}  // namespace ai_factory::cuda::rough_bergomi_terminal
