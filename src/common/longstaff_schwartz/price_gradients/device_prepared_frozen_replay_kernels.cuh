// Frozen-replay moments for device-prepared first and diagonal sensitivities.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <string>
#include <type_traits>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<pg::SensitivityOrders Orders>
inline constexpr std::size_t kSensitivityMomentCount =
    2U * (static_cast<std::size_t>(pg::requests_first_v<Orders>)
          + static_cast<std::size_t>(pg::requests_second_v<Orders>));

template<pg::SensitivityOrders Orders, typename Policy>
__global__ void frozen_replay_sensitivity_moments_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t sensitivity_count,
    const RegressionDiagnostics* diagnostics,
    std::size_t policy_workspace_bytes,
    double* partials
) {
    constexpr std::size_t moment_count =
        kSensitivityMomentCount<Orders>;
    const std::size_t task = blockIdx.y;
    const std::size_t batch_price = task / sensitivity_count;
    const std::size_t sensitivity = task % sensitivity_count;

    const std::size_t warp_count = (blockDim.x + 31U) / 32U;
    const std::size_t reduction_bytes =
        moment_count * (warp_count + 1U) * sizeof(double);
    extern __shared__ unsigned char dynamic_shared[];
    unsigned char* const policy_workspace =
        dynamic_shared + reduction_bytes;

    using PreparedSensitivity = typename Policy::PreparedSensitivity;
    static_assert(std::is_trivially_copyable_v<PreparedSensitivity>);
    static_assert(std::is_trivially_destructible_v<PreparedSensitivity>);
    __shared__ typename Policy::PreparedRow row;
    __shared__ alignas(PreparedSensitivity)
        unsigned char prepared_storage[sizeof(PreparedSensitivity)];
    PreparedSensitivity& prepared =
        *reinterpret_cast<PreparedSensitivity*>(prepared_storage);
    if (threadIdx.x == 0U) {
        row = rows[batch_price];
        if constexpr (requires {
            Policy::prepare_sensitivity(
                row,
                sensitivity,
                blockIdx.x == 0U,
                policy_workspace,
                policy_workspace_bytes
            );
        }) {
            prepared = Policy::prepare_sensitivity(
                row,
                sensitivity,
                blockIdx.x == 0U,
                policy_workspace,
                policy_workspace_bytes
            );
        } else {
            prepared = Policy::prepare_sensitivity(
                row, sensitivity, blockIdx.x == 0U
            );
        }
    }
    __syncthreads();

    double moments[moment_count]{};
    if (prepared.valid
        && diagnostics[batch_price].fatal_failure_count == 0U
        && (!Policy::kCanExerciseAtInitialTime
            || Policy::kFrozenRegressionPolicy
            || Policy::initial_exercise_decision(row)
                == longstaff_schwartz::InitialExerciseDecision::continuation)) {
        for (std::size_t path =
                 static_cast<std::size_t>(blockIdx.x) * blockDim.x
                     + threadIdx.x;
             path < paths_per_price;
             path += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
            const pg::SensitivityResult value =
                Policy::path_sensitivity(row, prepared, path);
            std::size_t channel = 0U;
            if constexpr (pg::requests_first_v<Orders>) {
                const double first = static_cast<double>(value.first);
                moments[channel++] += first;
                moments[channel++] += first * first;
            }
            if constexpr (pg::requests_second_v<Orders>) {
                const double second = static_cast<double>(value.second);
                moments[channel++] += second;
                moments[channel++] += second * second;
            }
        }
    }
    const double* totals = reductions::reduce_block_values(moments);
    if (threadIdx.x == 0U) {
        #pragma unroll
        for (std::size_t channel = 0U;
             channel < moment_count;
             ++channel) {
            partials[(task * moment_count + channel) * blocks_per_price
                     + blockIdx.x] = totals[channel];
        }
    }
}

template<pg::SensitivityOrders Orders, typename Policy>
__global__ void finalize_frozen_replay_sensitivities_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t sensitivity_count,
    const RegressionDiagnostics* diagnostics,
    const double* partials,
    pg::SensitivityOutputs outputs
) {
    constexpr std::size_t moment_count =
        kSensitivityMomentCount<Orders>;
    const std::size_t task = blockIdx.x;
    const std::size_t batch_price = task / sensitivity_count;
    const std::size_t sensitivity = task % sensitivity_count;
    const auto& row = rows[batch_price];
    const std::size_t output = row.result_index * sensitivity_count
        + sensitivity;

    const auto invalidate = [&] {
        if (threadIdx.x != 0U) return;
        if constexpr (pg::requests_first_v<Orders>) {
            outputs.gradients[output] = nanf("");
            outputs.gradient_standard_errors[output] = nanf("");
        }
        if constexpr (pg::requests_second_v<Orders>) {
            outputs.diagonal_hessians[output] = nanf("");
            outputs.diagonal_hessian_standard_errors[output] = nanf("");
        }
    };
    if (diagnostics[batch_price].fatal_failure_count != 0U
        || Policy::initial_exercise_decision(row)
            == longstaff_schwartz::InitialExerciseDecision::invalid) {
        invalidate();
        return;
    }

    if constexpr (Policy::kCanExerciseAtInitialTime
                  && !Policy::kFrozenRegressionPolicy) {
        if (Policy::initial_exercise_decision(row)
            == longstaff_schwartz::InitialExerciseDecision::exercise) {
            using PreparedSensitivity = typename Policy::PreparedSensitivity;
            static_assert(std::is_trivially_copyable_v<PreparedSensitivity>);
            static_assert(std::is_trivially_destructible_v<PreparedSensitivity>);
            __shared__ alignas(PreparedSensitivity)
                unsigned char initial_storage[sizeof(PreparedSensitivity)];
            PreparedSensitivity& initial_prepared =
                *reinterpret_cast<PreparedSensitivity*>(initial_storage);
            if (threadIdx.x == 0U) {
                initial_prepared = Policy::prepare_sensitivity(
                    row, sensitivity, false
                );
            }
            __syncthreads();
            if (!initial_prepared.valid) {
                invalidate();
                return;
            }
            if (threadIdx.x == 0U) {
                const pg::SensitivityResult value =
                    Policy::initial_sensitivity(row, initial_prepared);
                if constexpr (pg::requests_first_v<Orders>) {
                    outputs.gradients[output] = value.first;
                    outputs.gradient_standard_errors[output] = 0.0f;
                }
                if constexpr (pg::requests_second_v<Orders>) {
                    outputs.diagonal_hessians[output] = value.second;
                    outputs.diagonal_hessian_standard_errors[output] = 0.0f;
                }
            }
            return;
        }
    }

    double moments[moment_count]{};
    for (std::size_t block = threadIdx.x;
         block < blocks_per_price;
         block += blockDim.x) {
        #pragma unroll
        for (std::size_t channel = 0U;
             channel < moment_count;
             ++channel) {
            moments[channel] += partials[
                (task * moment_count + channel) * blocks_per_price + block
            ];
        }
    }
    const double* totals = reductions::reduce_block_values(moments);
    if (threadIdx.x != 0U) return;

    std::size_t channel = 0U;
    const auto publish = [&](float* values, float* errors) {
        double mean = 0.0;
        double error = 0.0;
        reductions::compute_statistics(
            {totals[channel], totals[channel + 1U]},
            paths_per_price,
            mean,
            error
        );
        values[output] = static_cast<float>(mean);
        errors[output] = static_cast<float>(error);
        channel += 2U;
    };
    if constexpr (pg::requests_first_v<Orders>) {
        publish(outputs.gradients, outputs.gradient_standard_errors);
    }
    if constexpr (pg::requests_second_v<Orders>) {
        publish(
            outputs.diagonal_hessians,
            outputs.diagonal_hessian_standard_errors
        );
    }
}

template<pg::SensitivityOrders Orders, typename Policy>
std::size_t finish_device_prepared_frozen_replay_sensitivity_batch(
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
    constexpr std::size_t moment_count =
        kSensitivityMomentCount<Orders>;
    const std::size_t warp_count = (threads + 31U) / 32U;
    const std::size_t reduction_shared =
        moment_count * (warp_count + 1U) * sizeof(double);
    const std::size_t policy_workspace =
        Policy::sensitivity_workspace_bytes(inputs);
    const std::size_t shared = reduction_shared + policy_workspace;
    const std::string moments_name =
        std::string(name) + ".frozen_replay_sensitivity_moments";
    report_cuda_kernel_launch_if_enabled(
        moments_name.c_str(),
        pg::requests_second_v<Orders> ? "nodes=4" : "nodes=3",
        frozen_replay_sensitivity_moments_kernel<Orders, Policy>,
        moments_grid,
        dim3(threads),
        shared
    );
    frozen_replay_sensitivity_moments_kernel<Orders, Policy><<<
        moments_grid, threads, shared
    >>>(
        rows,
        paths,
        blocks,
        inputs.sensitivity_count,
        diagnostics,
        policy_workspace,
        partials
    );
    check_cuda(cudaGetLastError(), "reduce frozen-replay sensitivities");

    const std::string finalize_name =
        std::string(name) + ".finalize_frozen_replay_sensitivities";
    report_cuda_kernel_launch_if_enabled(
        finalize_name.c_str(),
        pg::requests_second_v<Orders> ? "nodes=4" : "nodes=3",
        finalize_frozen_replay_sensitivities_kernel<Orders, Policy>,
        dim3(static_cast<unsigned int>(task_count)),
        dim3(threads),
        reduction_shared
    );
    finalize_frozen_replay_sensitivities_kernel<Orders, Policy><<<
        static_cast<unsigned int>(task_count), threads, reduction_shared
    >>>(
        rows,
        paths,
        blocks,
        inputs.sensitivity_count,
        diagnostics,
        partials,
        inputs.outputs
    );
    check_cuda(cudaGetLastError(), "finalize frozen-replay sensitivities");
    return 2U;
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
