// Paired selected-gradient moments after one central frozen-exercise LSM solve.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/longstaff_schwartz/price_gradients/execution_plan.cuh"

#include <string>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

template<typename Policy>
__global__ void frozen_gradient_moments_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t sensitivity_count,
    const RegressionDiagnostics* diagnostics,
    double* partials
) {
    const std::size_t task = blockIdx.y;
    const std::size_t batch_price = task / sensitivity_count;
    const std::size_t sensitivity = task % sensitivity_count;

    __shared__ typename Policy::PreparedRow row;
    __shared__ typename Policy::PreparedSensitivity prepared;
    if (threadIdx.x == 0U) {
        row = rows[batch_price];
        prepared = Policy::prepare_sensitivity(row, sensitivity);
    }
    __syncthreads();

    double sum = 0.0;
    double sumsq = 0.0;
    if (diagnostics[batch_price].fatal_failure_count == 0U
        && *row.initial_exercise
            == longstaff_schwartz::InitialExerciseDecision::continuation) {
        for (std::size_t path =
                 static_cast<std::size_t>(blockIdx.x) * blockDim.x
                     + threadIdx.x;
             path < paths_per_price;
             path += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
            const double gradient = static_cast<double>(
                Policy::path_gradient(row, prepared, path)
            );
            sum += gradient;
            sumsq += gradient * gradient;
        }
    }
    const auto totals = reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        partials[(task * 2U) * blocks_per_price + blockIdx.x] =
            totals.sum;
        partials[(task * 2U + 1U) * blocks_per_price + blockIdx.x] =
            totals.sumsq;
    }
}

template<typename Policy>
__global__ void finalize_frozen_gradients_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t sensitivity_count,
    const RegressionDiagnostics* diagnostics,
    const double* partials,
    float* gradients,
    float* standard_errors
) {
    const std::size_t task = blockIdx.x;
    const std::size_t batch_price = task / sensitivity_count;
    const std::size_t sensitivity = task % sensitivity_count;
    const auto& row = rows[batch_price];
    const std::size_t output = row.result_index * sensitivity_count
        + sensitivity;

    if (diagnostics[batch_price].fatal_failure_count != 0U) {
        if (threadIdx.x == 0U) {
            gradients[output] = nanf("");
            standard_errors[output] = nanf("");
        }
        return;
    }
    if (*row.initial_exercise
        == longstaff_schwartz::InitialExerciseDecision::invalid) {
        if (threadIdx.x == 0U) {
            gradients[output] = nanf("");
            standard_errors[output] = nanf("");
        }
        return;
    }
    if (*row.initial_exercise
        == longstaff_schwartz::InitialExerciseDecision::exercise) {
        if (threadIdx.x == 0U) {
            const auto prepared = Policy::prepare_sensitivity(
                row, sensitivity
            );
            gradients[output] = Policy::initial_gradient(row, prepared);
            standard_errors[output] = 0.0f;
        }
        return;
    }

    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t block = threadIdx.x;
         block < blocks_per_price;
         block += blockDim.x) {
        sum += partials[(task * 2U) * blocks_per_price + block];
        sumsq += partials[(task * 2U + 1U) * blocks_per_price + block];
    }
    const auto totals = reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        double gradient = 0.0;
        double error = 0.0;
        reductions::compute_statistics(
            totals, paths_per_price, gradient, error
        );
        gradients[output] = static_cast<float>(gradient);
        standard_errors[output] = static_cast<float>(error);
    }
}

template<typename Policy>
std::size_t finish_frozen_gradient_batch(
    const typename Policy::DeviceInputs& inputs,
    const typename Policy::PreparedRow* rows,
    dim3 central_grid,
    unsigned int threads,
    std::size_t paths,
    std::size_t blocks,
    double* partials,
    const RegressionDiagnostics* diagnostics,
    const char* name
) {
    if (inputs.sensitivity_count == 0U) return 0U;
    const std::size_t task_count =
        static_cast<std::size_t>(central_grid.y)
        * inputs.sensitivity_count;
    const dim3 moments_grid(
        central_grid.x,
        static_cast<unsigned int>(task_count)
    );
    const std::size_t shared =
        2U * (threads / 32U) * sizeof(double);
    const std::string moments_name =
        std::string(name) + ".frozen_gradient_moments";
    report_cuda_kernel_launch_if_enabled(
        moments_name.c_str(),
        "paired",
        frozen_gradient_moments_kernel<Policy>,
        moments_grid,
        dim3(threads),
        shared
    );
    frozen_gradient_moments_kernel<Policy><<<
        moments_grid, threads, shared
    >>>(
        rows,
        paths,
        blocks,
        inputs.sensitivity_count,
        diagnostics,
        partials
    );
    check_cuda(cudaGetLastError(), "reduce frozen-exercise gradients");

    const std::string finalize_name =
        std::string(name) + ".finalize_frozen_gradients";
    report_cuda_kernel_launch_if_enabled(
        finalize_name.c_str(),
        "paired",
        finalize_frozen_gradients_kernel<Policy>,
        dim3(static_cast<unsigned int>(task_count)),
        dim3(threads),
        shared
    );
    finalize_frozen_gradients_kernel<Policy><<<
        static_cast<unsigned int>(task_count), threads, shared
    >>>(
        rows,
        paths,
        blocks,
        inputs.sensitivity_count,
        diagnostics,
        partials,
        inputs.gradients,
        inputs.gradient_errors
    );
    check_cuda(cudaGetLastError(), "finalize frozen-exercise gradients");
    return 2U;
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
