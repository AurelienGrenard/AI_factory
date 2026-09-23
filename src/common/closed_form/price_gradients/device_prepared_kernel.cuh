// One thread prepares and evaluates one analytical price with selected sensitivities.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/price_gradients/device_preparation.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"
#include "common/price_gradients/reconstruction.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::closed_form::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Preparation
>
__global__ void device_prepared_kernel(
    mcpg::DevicePreparedInputs<Preparation> inputs,
    mcpg::DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs
) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t local_row =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         local_row < launch.result_count;
         local_row += stride) {
        const std::size_t row = launch.result_offset + local_row;
        const auto indices = pg::price_row_indices(
            row, plan.construction, plan.product_count
        );
        typename Preparation::Scenario central{};
        if (!Preparation::make_central(
                inputs.models[indices.model],
                inputs.products[indices.product],
                plan.time,
                central
            )) {
            preparation::record_error(
                stencil_outputs.error,
                preparation::invalid_central,
                row,
                0U
            );
            continue;
        }
        const float central_value = Policy::evaluate(central);
        outputs.prices[row] = central_value;
        for (std::size_t sensitivity = 0U;
             sensitivity < plan.sensitivity_count;
             ++sensitivity) {
            pg::SensitivityTask<
                typename Preparation::Scenario,
                node_capacity
            > task{};
            int error = preparation::valid;
            if (!preparation::build_sensitivity_task<Orders, Preparation>(
                    central,
                    inputs.sensitivities[sensitivity],
                    plan.time,
                    task,
                    error
                )) {
                preparation::record_error(
                    stencil_outputs.error,
                    error,
                    row,
                    sensitivity
                );
                continue;
            }
            pg::SensitivityValues<node_capacity> values{};
            values[0U] = central_value;
            for (std::size_t node = 1U;
                 node < pg::active_node_count(task.stencil);
                 ++node) {
                values[node] = Policy::evaluate(task.nodes[node]);
            }
            const auto result = pg::reconstruct_sensitivity<Orders>(
                task.stencil, values
            );
            const auto output = row * plan.sensitivity_count + sensitivity;
            if constexpr (pg::requests_first_v<Orders>) {
                outputs.gradients[output] = result.first;
            }
            if constexpr (pg::requests_second_v<Orders>) {
                outputs.diagonal_hessians[output] = result.second;
            }
            stencil_outputs.stencils[output] = task.stencil;
        }
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename HostPlan
>
void launch_device_prepared(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    const char* kernel_name,
    const char* variant
) {
    static_assert(Orders != pg::SensitivityOrders::none);
    if (host.request.orders != Orders) {
        throw std::invalid_argument(
            "Sensitivity request and analytical kernel order differ."
        );
    }
    const auto rows = host.result_count;
    const auto sensitivity_count = rows * host.sensitivity_count();
    if (outputs.prices == nullptr || outputs.price_capacity < rows) {
        throw std::invalid_argument("Insufficient analytical price outputs.");
    }
    std::vector<pg::BufferRange> output_ranges{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
    };
    if constexpr (pg::requests_first_v<Orders>) {
        if (host.sensitivity_count() != 0U
            && (outputs.gradients == nullptr
                || outputs.sensitivity_capacity < sensitivity_count)) {
            throw std::invalid_argument(
                "Insufficient analytical gradient outputs."
            );
        }
        if (host.sensitivity_count() != 0U) {
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.gradients, sensitivity_count, sizeof(float)
            ));
        }
    }
    if constexpr (pg::requests_second_v<Orders>) {
        if (host.sensitivity_count() != 0U
            && (outputs.diagonal_hessians == nullptr
                || outputs.sensitivity_capacity < sensitivity_count)) {
            throw std::invalid_argument(
                "Insufficient analytical diagonal-Hessian outputs."
            );
        }
        if (host.sensitivity_count() != 0U) {
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.diagonal_hessians,
                sensitivity_count,
                sizeof(float)
            ));
        }
    }
    const auto plan = mcpg::make_device_prepared_plan(host);
    mcpg::validate_device_prepared_launch(
        device,
        plan,
        stencil_outputs,
        configuration,
        output_ranges
    );
    const auto function = device_prepared_kernel<
        Orders,
        Policy,
        typename HostPlan::Preparation
    >;
    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        variant,
        function,
        dim3(static_cast<unsigned int>(configuration.block_count)),
        dim3(configuration.threads_per_block),
        0U
    );
    function<<<configuration.block_count, configuration.threads_per_block>>>(
        device,
        plan,
        configuration,
        outputs,
        stencil_outputs
    );
    check_cuda(cudaGetLastError(), kernel_name);
}

}  // namespace ai_factory::workbench::closed_form::price_gradients
