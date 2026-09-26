// Build one row's central node, unique sensitivity nodes and reconstruction map.
#pragma once

#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

template<
    pg::SensitivityOrders Orders,
    typename Preparation,
    std::size_t MaximumSensitivities>
__device__ __forceinline__ bool prepare_sensitivity_row_from_central(
    const typename Preparation::Scenario& central,
    const pg::SensitivitySpec<typename Preparation::Parameter>* sensitivities,
    std::size_t sensitivity_count,
    pg::TimeConfiguration time,
    typename Preparation::Scenario* scenarios,
    pg::SensitivityStencil<4U>* stencils,
    SensitivityNodeIndices<4U>* node_indices,
    std::uint16_t& node_count,
    std::uint32_t& maximum_steps,
    int& error,
    std::size_t& error_sensitivity
) {
    static_assert(pg::requests_second_v<Orders>);
    scenarios[0U] = central;
    node_count = 1U;
    maximum_steps = central.step_count;
    for (std::size_t sensitivity = 0U;
         sensitivity < sensitivity_count;
         ++sensitivity) {
        pg::SensitivityTask<typename Preparation::Scenario, 4U> task{};
        if (!build_terminal_sensitivity_task_from_central<
                Orders, Preparation
            >(
                central,
                sensitivities[sensitivity],
                time,
                task,
                error
            )) {
            error_sensitivity = sensitivity;
            return false;
        }
        stencils[sensitivity] = task.stencil;
        auto& indices = node_indices[sensitivity];
        indices[0U] = 0U;
        const auto active = pg::active_node_count(task.stencil);
        for (std::size_t local_node = 1U;
             local_node < active;
             ++local_node) {
            if (node_count
                >= terminal_node_graph_node_capacity<
                    MaximumSensitivities
                >()) {
                error = preparation::unsupported_order;
                error_sensitivity = sensitivity;
                return false;
            }
            indices[local_node] = node_count;
            scenarios[node_count] = task.nodes[local_node];
            maximum_steps = max(
                maximum_steps, task.nodes[local_node].step_count
            );
            ++node_count;
        }
    }
    return true;
}

template<
    pg::SensitivityOrders Orders,
    typename Inputs,
    typename Preparation,
    std::size_t MaximumSensitivities>
__device__ __forceinline__ bool prepare_sensitivity_row(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t row,
    typename Preparation::Scenario* scenarios,
    pg::SensitivityStencil<4U>* stencils,
    SensitivityNodeIndices<4U>* node_indices,
    std::uint16_t& node_count,
    std::uint32_t& maximum_steps,
    int& error,
    std::size_t& error_sensitivity
) {
    static_assert(pg::requests_second_v<Orders>);
    typename Preparation::Scenario central{};
    if (!inputs.make_central(row, plan, central)) {
        error = preparation::invalid_central;
        error_sensitivity = 0U;
        return false;
    }

    return prepare_sensitivity_row_from_central<
        Orders, Preparation, MaximumSensitivities
    >(
        central,
        inputs.sensitivities,
        plan.sensitivity_count,
        plan.time,
        scenarios,
        stencils,
        node_indices,
        node_count,
        maximum_steps,
        error,
        error_sensitivity
    );
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
