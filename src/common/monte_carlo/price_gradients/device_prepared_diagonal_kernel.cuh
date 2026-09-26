// Generic one-block-per-(row,sensitivity) diagonal Monte Carlo kernel.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"
#include "common/philox.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::equity::price_gradients::device_preparation;

namespace diagonal_sensitivity_detail {

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Preparation,
    typename Inputs>
__device__ __noinline__ bool prepare_row(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t row,
    std::size_t sensitivity_index,
    typename Policy::Nodes* nodes,
    typename Policy::Stencil* stencil,
    typename Policy::PreparedRow* prepared,
    DevicePreparedStencilOutputs<4U> stencil_outputs
) {
    int preparation_error = preparation::valid;
    pg::SensitivityTask<typename Preparation::Scenario, 4U> task{};
    const bool valid_row =
        build_terminal_sensitivity_task<Orders, decltype(inputs), Preparation>(
            inputs,
            plan,
            row,
            sensitivity_index,
            task,
            preparation_error
        );
    if (valid_row) {
        *nodes = task.nodes;
        *stencil = task.stencil;
        *prepared = Policy::prepare(nodes, stencil, plan.time);
        stencil_outputs.stencils[
            row * plan.sensitivity_count + sensitivity_index
        ] = *stencil;
    } else {
        preparation::record_error(
            stencil_outputs.error,
            preparation_error,
            row,
            sensitivity_index
        );
    }
    return valid_row;
}

template<typename Policy, pg::SensitivityOrders Orders, bool OwnsPrice>
__device__ __forceinline__ void evaluate_and_reduce(
    const typename Policy::PreparedRow& prepared,
    philox::PhiloxKey key,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    std::size_t row,
    std::size_t sensitivity_index,
    std::size_t sensitivity_count
) {
    constexpr unsigned int price_channels = OwnsPrice ? 1U : 0U;
    constexpr unsigned int first_channels =
        pg::requests_first_v<Orders> ? 1U : 0U;
    constexpr unsigned int second_channels =
        pg::requests_second_v<Orders> ? 1U : 0U;
    constexpr unsigned int channel_count =
        price_channels + first_channels + second_channels;
    double sums[channel_count]{};
    double squares[channel_count]{};
    for (std::size_t path = threadIdx.x;
         path < launch.paths_per_price;
         path += blockDim.x) {
        const auto result = Policy::evaluate_path(prepared, key, path);
        unsigned int channel = 0U;
        if constexpr (OwnsPrice) {
            const double price = static_cast<double>(result.price);
            sums[channel] += price;
            squares[channel] += price * price;
            ++channel;
        }
        if constexpr (pg::requests_first_v<Orders>) {
            const double first = static_cast<double>(result.first);
            sums[channel] += first;
            squares[channel] += first * first;
            ++channel;
        }
        if constexpr (pg::requests_second_v<Orders>) {
            const double second = static_cast<double>(result.second);
            sums[channel] += second;
            squares[channel] += second * second;
        }
    }
    #pragma unroll
    for (unsigned int channel = 0U;
         channel < channel_count;
         ++channel) {
        const auto moments = reductions::reduce_block(
            sums[channel], squares[channel]
        );
        if (threadIdx.x == 0U) {
            double value = 0.0;
            double error = 0.0;
            reductions::compute_statistics(
                moments,
                launch.paths_per_price,
                value,
                error,
                launch.paths_per_price / blockDim.x
                    + (launch.paths_per_price % blockDim.x != 0U)
            );
            unsigned int output_channel = 0U;
            if constexpr (OwnsPrice) {
                if (channel == output_channel) {
                    outputs.prices[row] = static_cast<float>(value);
                    outputs.price_standard_errors[row] =
                        static_cast<float>(error);
                }
                ++output_channel;
            }
            const auto index = row * sensitivity_count + sensitivity_index;
            if constexpr (pg::requests_first_v<Orders>) {
                if (channel == output_channel) {
                    outputs.gradients[index] = static_cast<float>(value);
                    outputs.gradient_standard_errors[index] =
                        static_cast<float>(error);
                }
                ++output_channel;
            }
            if constexpr (pg::requests_second_v<Orders>) {
                if (channel == output_channel) {
                    outputs.diagonal_hessians[index] =
                        static_cast<float>(value);
                    outputs.diagonal_hessian_standard_errors[index] =
                        static_cast<float>(error);
                }
            }
        }
        __syncthreads();
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Preparation,
    typename Inputs>
__device__ __forceinline__ void kernel_body(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    __shared__ typename Policy::Nodes nodes;
    __shared__ typename Policy::Stencil stencil;
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;
    __shared__ bool valid_row;

    const std::size_t sensitivity_index = blockIdx.y;
    if (sensitivity_index >= plan.sensitivity_count) return;
    const bool owns_price = sensitivity_index == 0U;
    for (std::size_t local_row = blockIdx.x;
         local_row < launch.result_count;
         local_row += gridDim.x) {
        const std::size_t row = launch.result_offset + local_row;
        if (threadIdx.x == 0U) {
            valid_row = prepare_row<Orders, Policy, Preparation>(
                inputs,
                plan,
                row,
                sensitivity_index,
                &nodes,
                &stencil,
                &prepared,
                stencil_outputs
            );
            if (valid_row) key = philox::make_key(base_seed + row);
        }
        __syncthreads();
        if (!valid_row) {
            __syncthreads();
            continue;
        }
        if (owns_price) {
            evaluate_and_reduce<Policy, Orders, true>(
                prepared,
                key,
                launch,
                outputs,
                row,
                sensitivity_index,
                plan.sensitivity_count
            );
        } else {
            evaluate_and_reduce<Policy, Orders, false>(
                prepared,
                key,
                launch,
                outputs,
                row,
                sensitivity_index,
                plan.sensitivity_count
            );
        }
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Preparation,
    typename Inputs>
__global__ void kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    kernel_body<Orders, Policy, Preparation, Inputs>(
        inputs, plan, launch, outputs, stencil_outputs, base_seed
    );
}

template<
    typename Tuning,
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Preparation,
    typename Inputs>
__global__ __launch_bounds__(
    Tuning::kThreadsPerBlock,
    Tuning::kMinimumBlocksPerMultiprocessor
) void bounded_kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    kernel_body<Orders, Policy, Preparation, Inputs>(
        inputs, plan, launch, outputs, stencil_outputs, base_seed
    );
}

}  // namespace diagonal_sensitivity_detail

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Preparation,
    typename Tuning = tuning::DefaultTerminalMonoTuning,
    typename Inputs>
void launch_device_prepared_diagonal_sensitivities(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
    static_assert(pg::requests_second_v<Orders>);
    if (plan.sensitivity_count == 0U) {
        throw std::invalid_argument(
            "A device-prepared sensitivity launch requires a selection."
        );
    }
    if (configuration.block_count < plan.sensitivity_count) {
        throw std::invalid_argument(
            "The block cap cannot cover every selected sensitivity."
        );
    }
    const std::size_t x_blocks = std::min(
        configuration.result_count,
        configuration.block_count / plan.sensitivity_count
    );
    const std::size_t shared = 2U
        * (configuration.threads_per_block / 32U) * sizeof(double);
    static_assert(tuning::valid_profile_v<Tuning>);
    using KernelFunction = void (*)(
        Inputs,
        DevicePreparedPlan,
        pg::LaunchConfiguration,
        pg::SensitivityOutputs,
        DevicePreparedStencilOutputs<4U>,
        std::uint64_t
    );
    KernelFunction function =
        diagonal_sensitivity_detail::kernel<
            Orders, Policy, Preparation, Inputs
        >;
    if constexpr (Tuning::kLaunchBoundsEnabled) {
        if (configuration.threads_per_block == Tuning::kThreadsPerBlock) {
            function = diagonal_sensitivity_detail::bounded_kernel<
                Tuning, Orders, Policy, Preparation, Inputs
            >;
        }
    }
    const dim3 grid(
        static_cast<unsigned int>(x_blocks),
        static_cast<unsigned int>(plan.sensitivity_count)
    );
    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        variant,
        function,
        grid,
        dim3(configuration.threads_per_block),
        shared
    );
    function<<<grid, configuration.threads_per_block, shared>>>(
        inputs,
        plan,
        configuration,
        outputs,
        stencil_outputs,
        configuration.base_seed
    );
    check_cuda(cudaGetLastError(), kernel_name);
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
