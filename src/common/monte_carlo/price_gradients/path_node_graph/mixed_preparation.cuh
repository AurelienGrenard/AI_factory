// Prepare one row of mixed path nodes once for every path chunk and shard.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/mixed_row_preparation.cuh"
#include "common/monte_carlo/price_gradients/path_node_graph/mixed_workspace.cuh"
#include "common/monte_carlo/price_gradients/path_sensitivity_policy.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil_outputs.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace path_node_graph_detail {

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    bool VariableTerminal,
    typename Inputs>
__global__ void prepare_mixed_rows_kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph sensitivity_graph,
    std::size_t first_row,
    std::size_t row_count,
    MixedPathNodeGraphWorkspace<
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>,
        Dynamics,
        Schedule
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs
) {
    using NodePolicy =
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>;
    using Scenario = typename Preparation::Scenario;
    using Traits = PathScheduleTraits<Schedule>;
    constexpr auto interval_capacity =
        path_dynamics_interval_capacity_v<Schedule>;

    __shared__ pg::SensitivityStencil<4U>
        stencils[MaximumSensitivities];
    __shared__ SensitivityNodeIndices<4U>
        axis_node_indices[MaximumSensitivities];
    __shared__ pg::MixedSensitivityStencil
        mixed_stencils[MaximumMixedSensitivities];
    __shared__ MixedSensitivityNodeIndices
        mixed_node_indices[MaximumMixedSensitivities];
    __shared__ std::uint16_t node_count;
    __shared__ std::uint32_t maximum_steps;
    __shared__ bool valid_row;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* scenarios = reinterpret_cast<Scenario*>(dynamic_shared);

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count) return;
    const std::size_t row = first_row + local_row;

    if (threadIdx.x == 0U) {
        int error = preparation::valid;
        std::size_t error_sensitivity = 0U;
        valid_row = node_graph_detail::prepare_mixed_sensitivity_row<
            Inputs,
            Preparation,
            MaximumSensitivities,
            MaximumMixedSensitivities
        >(
            inputs,
            plan,
            sensitivity_graph,
            row,
            scenarios,
            stencils,
            axis_node_indices,
            mixed_stencils,
            mixed_node_indices,
            node_count,
            maximum_steps,
            error,
            error_sensitivity
        );
        workspace.graph.row_status[local_row] =
            static_cast<std::uint8_t>(valid_row);
        if (valid_row) {
            workspace.node_counts[local_row] = node_count;
            const auto calendar =
                ProductPolicy::calendar(scenarios[0U].product);
            if constexpr (VariableTerminal) {
                workspace.schedules[local_row] =
                    prepare_terminal_path_schedule<Schedule>(
                        calendar,
                        plan.time,
                        scenarios[0U].central_step_count
                    );
                workspace.maximum_terminal_transitions[local_row] = 0U;
            } else {
                workspace.schedules[local_row].central =
                    prepare_path_schedule<Schedule>(calendar, plan.time);
            }
            for (std::size_t sensitivity = 0U;
                 sensitivity < plan.sensitivity_count;
                 ++sensitivity) {
                stencil_outputs.stencils[
                    row * plan.sensitivity_count + sensitivity
                ] = stencils[sensitivity];
                workspace.graph.axis_node_indices[
                    local_row * plan.sensitivity_count + sensitivity
                ] = axis_node_indices[sensitivity];
            }
            for (std::size_t pair = 0U;
                 pair < sensitivity_graph.mixed_second_count;
                 ++pair) {
                mixed_stencil_outputs.stencils[
                    row * sensitivity_graph.mixed_second_count + pair
                ] = mixed_stencils[pair];
                workspace.graph.mixed_node_indices[
                    local_row * sensitivity_graph.mixed_second_count + pair
                ] = mixed_node_indices[pair];
            }
        } else {
            preparation::record_error(
                stencil_outputs.error,
                error,
                row,
                error_sensitivity
            );
        }
    }
    __syncthreads();
    if (!valid_row) return;

    const auto& terminal_schedule = workspace.schedules[local_row];
    const auto& schedule = terminal_schedule.central;
    for (std::size_t node = threadIdx.x;
         node < node_count;
         node += blockDim.x) {
        workspace.graph.node_metadata[
            local_row * sensitivity_graph.node_capacity + node
        ] = NodePolicy::prepare_metadata(scenarios[node], plan.time);
        workspace.simulation_flags[
            local_row * sensitivity_graph.node_capacity + node
        ] = static_cast<std::uint8_t>(
            node == 0U || !scenarios[node].reuse_central
        );
        #pragma unroll
        for (std::size_t interval = 0U;
             interval < interval_capacity;
             ++interval) {
            const float horizon = Traits::kExactTransition
                ? schedule.interval_years[interval]
                : plan.time.dt;
            workspace.prepared_dynamics[
                (local_row * interval_capacity + interval)
                    * sensitivity_graph.node_capacity
                + node
            ] = NodePolicy::prepare_dynamics(scenarios[node], horizon);
        }
        if constexpr (VariableTerminal) {
            const auto terminal_node = prepare_terminal_path_node<Schedule>(
                scenarios[node], terminal_schedule
            );
            const auto row_node =
                local_row * sensitivity_graph.node_capacity + node;
            workspace.terminal_transition_counts[row_node] =
                terminal_node.transition_count;
            const float terminal_horizon = Traits::kExactTransition
                ? terminal_node.interval_years
                : plan.time.dt;
            workspace.terminal_dynamics[row_node] =
                NodePolicy::prepare_dynamics(
                    scenarios[node], terminal_horizon
                );
            #pragma unroll
            for (unsigned int weight = 0U; weight < 4U; ++weight) {
                workspace.terminal_normal_weights[
                    row_node * 4U + weight
                ] = scenarios[node].normal_weights[weight];
            }
            atomicMax(
                workspace.maximum_terminal_transitions + local_row,
                terminal_node.transition_count
            );
        }
    }
}

}  // namespace path_node_graph_detail

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
