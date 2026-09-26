// Generic one-block-per-(row,sensitivity) first-order Monte Carlo kernel.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"
#include "common/philox.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/launch.cuh"
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

namespace first_sensitivity_detail {

template<typename Policy, typename Preparation, typename Inputs>
__device__ __noinline__ int prepare_row(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t row,
    std::size_t sensitivity_index,
    typename Policy::Nodes* nodes,
    typename Policy::Stencil* stencil,
    typename Policy::PreparedRow* prepared,
    pg::CentralRequirement* central_requirement,
    bool computes_central_price,
    DevicePreparedStencilOutputs<3U> stencil_outputs
) {
    int preparation_error = preparation::valid;
    pg::SensitivityTask<typename Preparation::Scenario, 3U> task{};
    const bool row_valid = build_terminal_sensitivity_task<
        pg::SensitivityOrders::first,
        decltype(inputs),
        Preparation
    >(
        inputs,
        plan,
        row,
        sensitivity_index,
        task,
        preparation_error
    );
    if (row_valid) {
        *nodes = task.nodes;
        *stencil = task.stencil;
        *central_requirement = computes_central_price
            ? pg::CentralRequirement::payoff
            : task.central_requirement;
        *prepared = Policy::prepare(
            nodes,
            stencil,
            plan.time,
            *central_requirement != pg::CentralRequirement::none
        );
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
    return row_valid;
}

template<
    typename Policy,
    bool ComputesCentralPrice,
    bool IncludeCentral,
    bool IncludePayoff>
__device__ __forceinline__ void evaluate_and_reduce(
    const typename Policy::PreparedRow& prepared,
    philox::PhiloxKey key,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs,
    std::size_t row,
    std::size_t sensitivity_index,
    std::size_t sensitivity_count
) {
    constexpr unsigned int price_channels = ComputesCentralPrice ? 1U : 0U;
    constexpr unsigned int channel_count = 1U + price_channels;
    double sums[channel_count]{};
    double squares[channel_count]{};
    for (std::size_t path = threadIdx.x;
         path < launch.paths_per_price;
         path += blockDim.x) {
        const auto result = Policy::template evaluate_path<
            IncludeCentral, IncludePayoff
        >(prepared, key, path);
        if constexpr (ComputesCentralPrice) {
            const double price = static_cast<double>(result.price);
            sums[0U] += price;
            squares[0U] += price * price;
        }
        const double sensitivity = static_cast<double>(result.sensitivity);
        sums[price_channels] += sensitivity;
        squares[price_channels] += sensitivity * sensitivity;
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
            if constexpr (ComputesCentralPrice) {
                if (channel == 0U) {
                    outputs.prices[row] = static_cast<float>(value);
                    outputs.price_standard_errors[row] =
                        static_cast<float>(error);
                } else {
                    const auto index =
                        row * sensitivity_count + sensitivity_index;
                    outputs.gradients[index] = static_cast<float>(value);
                    outputs.gradient_standard_errors[index] =
                        static_cast<float>(error);
                }
            } else {
                const auto index =
                    row * sensitivity_count + sensitivity_index;
                outputs.gradients[index] = static_cast<float>(value);
                outputs.gradient_standard_errors[index] =
                    static_cast<float>(error);
            }
        }
        __syncthreads();
    }
}

template<typename Policy, typename Preparation, typename Inputs>
__global__ void kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::Outputs outputs,
    DevicePreparedStencilOutputs<3U> stencil_outputs,
    std::uint64_t base_seed
) {
    __shared__ typename Policy::Nodes nodes;
    __shared__ typename Policy::Stencil stencil;
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;
    __shared__ pg::CentralRequirement central_requirement;
    __shared__ int row_valid;

    const std::size_t sensitivity_index = blockIdx.y;
    if (sensitivity_index >= plan.sensitivity_count) return;
    const bool computes_central_price = sensitivity_index == 0U;
    for (std::size_t local_row = blockIdx.x;
         local_row < launch.result_count;
         local_row += gridDim.x) {
        const std::size_t row = launch.result_offset + local_row;
        if (threadIdx.x == 0U) {
            row_valid = prepare_row<Policy, Preparation>(
                inputs,
                plan,
                row,
                sensitivity_index,
                &nodes,
                &stencil,
                &prepared,
                &central_requirement,
                computes_central_price,
                stencil_outputs
            );
            if (row_valid) key = philox::make_key(base_seed + row);
        }
        __syncthreads();
        if (!row_valid) {
            __syncthreads();
            continue;
        }
        if (computes_central_price) {
            evaluate_and_reduce<Policy, true, true, true>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        } else if (central_requirement == pg::CentralRequirement::payoff) {
            evaluate_and_reduce<Policy, false, true, true>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        } else if (central_requirement == pg::CentralRequirement::state) {
            evaluate_and_reduce<Policy, false, true, false>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        } else {
            evaluate_and_reduce<Policy, false, false, false>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        }
    }
}

}  // namespace first_sensitivity_detail

template<typename Policy, typename Preparation, typename Inputs>
void launch_device_prepared_first_sensitivities(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs,
    DevicePreparedStencilOutputs<3U> stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
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
    const auto function =
        first_sensitivity_detail::kernel<Policy, Preparation, Inputs>;
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
