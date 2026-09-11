// Paired delta moments after one central LSM solve; reuses its completed workspace.
#pragma once

#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"

namespace ai_factory::workbench::longstaff_schwartz {

template<typename Policy>
__global__ void frozen_date_delta_moments_kernel(
    const typename Policy::PreparedRow* rows, std::size_t paths_per_price,
    std::size_t blocks_per_price, const RegressionDiagnostics* diagnostics,
    double* partials
) {
    __shared__ typename Policy::PreparedRow row;
    if (threadIdx.x == 0U) row = rows[blockIdx.y];
    __syncthreads();
    double sum = 0.0, sumsq = 0.0;
    if (diagnostics[blockIdx.y].fatal_failure_count == 0U
        && *row.initial_exercise == Policy::InitialDecision::continuation) {
        for (std::size_t path = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             path < paths_per_price; path += std::size_t(gridDim.x) * blockDim.x) {
            const double delta = Policy::path_delta(row, path);
            sum += delta;
            sumsq += delta * delta;
        }
    }
    const auto totals = reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        partials[(std::size_t(blockIdx.y) * 2U) * blocks_per_price + blockIdx.x] = totals.sum;
        partials[(std::size_t(blockIdx.y) * 2U + 1U) * blocks_per_price + blockIdx.x] = totals.sumsq;
    }
}

template<typename Policy>
__global__ void finalize_frozen_date_deltas_kernel(
    const typename Policy::PreparedRow* rows, std::size_t paths_per_price,
    std::size_t blocks_per_price, const RegressionDiagnostics* diagnostics,
    const double* partials, float* deltas, float* standard_errors
) {
    const auto& row = rows[blockIdx.x];
    if (diagnostics[blockIdx.x].fatal_failure_count != 0U) {
        if (threadIdx.x == 0U) invalidate_regression_result_if_fatal(
            diagnostics[blockIdx.x], row.result_index, deltas, standard_errors);
        return;
    }
    if (*row.initial_exercise == Policy::InitialDecision::invalid) {
        if (threadIdx.x == 0U) {
            deltas[row.result_index] = nanf("");
            standard_errors[row.result_index] = nanf("");
        }
        return;
    }
    if (*row.initial_exercise == Policy::InitialDecision::exercise) {
        if (threadIdx.x == 0U) {
            deltas[row.result_index] = Policy::initial_delta(row);
            standard_errors[row.result_index] = 0.0f;
        }
        return;
    }
    double sum = 0.0, sumsq = 0.0;
    for (std::size_t block = threadIdx.x; block < blocks_per_price; block += blockDim.x) {
        sum += partials[(std::size_t(blockIdx.x) * 2U) * blocks_per_price + block];
        sumsq += partials[(std::size_t(blockIdx.x) * 2U + 1U) * blocks_per_price + block];
    }
    const auto totals = reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        double delta = 0.0, error = 0.0;
        reductions::compute_statistics(totals, paths_per_price, delta, error);
        deltas[row.result_index] = static_cast<float>(delta);
        standard_errors[row.result_index] = static_cast<float>(error);
    }
}

template<typename Policy>
std::size_t finish_frozen_date_delta_batch(
    const typename Policy::DeviceInputs& inputs,
    const typename Policy::PreparedRow* rows, dim3 grid, unsigned threads,
    std::size_t paths, std::size_t blocks, double* partials,
    const RegressionDiagnostics* diagnostics, const char* name
) {
    const std::size_t shared = 2U * (threads / 32U) * sizeof(double);
    report_cuda_kernel_launch_if_enabled((std::string(name) + ".frozen_date_delta_moments").c_str(),
        "paired", frozen_date_delta_moments_kernel<Policy>, grid, dim3(threads), shared);
    frozen_date_delta_moments_kernel<Policy><<<grid, threads, shared>>>(
        rows, paths, blocks, diagnostics, partials);
    check_cuda(cudaGetLastError(), "reduce frozen-date delta moments");
    report_cuda_kernel_launch_if_enabled((std::string(name) + ".finalize_deltas").c_str(),
        "paired", finalize_frozen_date_deltas_kernel<Policy>, dim3(grid.y), dim3(threads), shared);
    finalize_frozen_date_deltas_kernel<Policy><<<grid.y, threads, shared>>>(
        rows, paths, blocks, diagnostics, partials, inputs.deltas, inputs.delta_errors);
    check_cuda(cudaGetLastError(), "finalize frozen-date deltas");
    return 2U;
}

}  // namespace ai_factory::workbench::longstaff_schwartz
