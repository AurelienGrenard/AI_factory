// One cooperative block prepares and evaluates one compact sensitivity row.
#pragma once

#include "common/closed_form/concepts.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"
#include "common/price_gradients/reconstruction.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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
__global__ void device_prepared_cooperative_kernel(
    mcpg::DevicePreparedInputs<Preparation> inputs,
    mcpg::DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs,
    std::uint32_t workspace_capacity
) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    using Scenario = typename Preparation::Scenario;
    using Task = pg::SensitivityTask<Scenario, node_capacity>;
    static_assert(
        sizeof(typename Policy::PreparedRow)
            <= closed_form::kMaximumSharedPreparedRowBytes,
        "Cooperative sensitivity PreparedRow exceeds the shared-row budget."
    );

    __shared__ Scenario central;
    __shared__ Task task;
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ pg::SensitivityValues<node_capacity> values;
    __shared__ bool valid;
    extern __shared__ __align__(16) unsigned char workspace_storage[];

    for (std::size_t launch_index = blockIdx.x;
         launch_index < launch.result_count;
         launch_index += gridDim.x) {
        const std::size_t row = launch.result_offset + launch_index;
        if (threadIdx.x == 0U) {
            const auto indices = pg::price_row_indices(
                row, plan.construction, plan.product_count
            );
            valid = Preparation::make_central(
                inputs.models[indices.model],
                inputs.products[indices.product],
                plan.time,
                central
            );
            if (valid) {
                prepared = Policy::prepare(central);
            } else {
                preparation::record_error(
                    stencil_outputs.error,
                    preparation::invalid_central,
                    row,
                    0U
                );
            }
        }
        __syncthreads();
        if (!valid) continue;

        const float central_value = Policy::evaluate(
            prepared,
            reinterpret_cast<std::byte*>(workspace_storage),
            workspace_capacity
        );
        if (threadIdx.x == 0U) {
            outputs.prices[row] = central_value;
        }
        __syncthreads();

        for (std::size_t sensitivity = 0U;
             sensitivity < plan.sensitivity_count;
             ++sensitivity) {
            if (threadIdx.x == 0U) {
                int error = preparation::valid;
                valid = preparation::build_sensitivity_task<
                    Orders,
                    Preparation
                >(
                    central,
                    inputs.sensitivities[sensitivity],
                    plan.time,
                    task,
                    error
                );
                if (!valid) {
                    preparation::record_error(
                        stencil_outputs.error,
                        error,
                        row,
                        sensitivity
                    );
                } else {
                    const std::size_t output =
                        row * plan.sensitivity_count + sensitivity;
                    stencil_outputs.stencils[output] = task.stencil;
                    values[0U] = central_value;
                }
            }
            __syncthreads();
            if (!valid) continue;

            const auto node_count = pg::active_node_count(task.stencil);
            for (std::size_t node = 1U; node < node_count; ++node) {
                if (threadIdx.x == 0U) {
                    prepared = Policy::prepare(task.nodes[node]);
                }
                __syncthreads();
                const float node_value = Policy::evaluate(
                    prepared,
                    reinterpret_cast<std::byte*>(workspace_storage),
                    workspace_capacity
                );
                if (threadIdx.x == 0U) values[node] = node_value;
                __syncthreads();
            }
            if (threadIdx.x == 0U) {
                const auto result = pg::reconstruct_sensitivity<Orders>(
                    task.stencil, values
                );
                const std::size_t output =
                    row * plan.sensitivity_count + sensitivity;
                if constexpr (pg::requests_first_v<Orders>) {
                    outputs.gradients[output] = result.first;
                }
                if constexpr (pg::requests_second_v<Orders>) {
                    outputs.diagonal_hessians[output] = result.second;
                }
            }
            __syncthreads();
        }
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename HostPlan
>
bool launch_device_prepared_cooperative(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    std::uint32_t workspace_capacity,
    const char* name,
    const char* variant
) {
    static_assert(Orders != pg::SensitivityOrders::none);
    if (host.request.orders != Orders) {
        throw std::invalid_argument(
            "Sensitivity request and cooperative kernel order differ."
        );
    }
    if (workspace_capacity == 0U) {
        throw std::invalid_argument(
            "Cooperative sensitivity workspace capacity is zero."
        );
    }
    const auto rows = host.result_count;
    const auto sensitivity_values = rows * host.sensitivity_count();
    if (outputs.prices == nullptr || outputs.price_capacity < rows) {
        throw std::invalid_argument("Insufficient analytical price outputs.");
    }
    std::vector<pg::BufferRange> output_ranges{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
    };
    if constexpr (pg::requests_first_v<Orders>) {
        if (host.sensitivity_count() != 0U
            && (outputs.gradients == nullptr
                || outputs.sensitivity_capacity < sensitivity_values)) {
            throw std::invalid_argument(
                "Insufficient cooperative gradient outputs."
            );
        }
        if (host.sensitivity_count() != 0U) {
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.gradients, sensitivity_values, sizeof(float)
            ));
        }
    }
    if constexpr (pg::requests_second_v<Orders>) {
        if (host.sensitivity_count() != 0U
            && (outputs.diagonal_hessians == nullptr
                || outputs.sensitivity_capacity < sensitivity_values)) {
            throw std::invalid_argument(
                "Insufficient cooperative diagonal-Hessian outputs."
            );
        }
        if (host.sensitivity_count() != 0U) {
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.diagonal_hessians,
                sensitivity_values,
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

    const auto function = device_prepared_cooperative_kernel<
        Orders,
        Policy,
        typename HostPlan::Preparation
    >;
    const std::size_t shared =
        Policy::required_shared_memory_bytes(workspace_capacity);
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, function), name);
    if (shared
        > static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes)) {
        return false;
    }
    int active = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            function,
            configuration.threads_per_block,
            shared
        ),
        name
    );
    if (active == 0) return false;

    const std::string diagnostic_variant = std::string(variant)
        + "/nodes="
        + std::to_string(pg::SensitivityTraits<Orders>::node_capacity)
        + "/K=" + std::to_string(host.sensitivity_count());
    report_cuda_kernel_launch_if_enabled(
        name,
        diagnostic_variant.c_str(),
        function,
        dim3(static_cast<unsigned int>(configuration.block_count)),
        dim3(configuration.threads_per_block),
        shared
    );
    function<<<configuration.block_count, configuration.threads_per_block,
               shared>>>(
        device,
        plan,
        configuration,
        outputs,
        stencil_outputs,
        workspace_capacity
    );
    check_cuda(cudaGetLastError(), name);
    return true;
}

}  // namespace ai_factory::workbench::closed_form::price_gradients
