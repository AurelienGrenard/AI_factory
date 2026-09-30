// Path-specific storage layered over the common mixed sensitivity node graph.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"
#include "common/monte_carlo/price_gradients/path_schedule.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::monte_carlo::price_gradients {

template<typename Schedule>
inline constexpr std::size_t path_dynamics_interval_capacity_v =
    PathScheduleTraits<Schedule>::kExactTransition
        ? PathScheduleTraits<Schedule>::kIntervalCapacity
        : 1U;

template<typename NodePolicy, typename Dynamics, typename Schedule>
struct MixedPathNodeGraphWorkspace {
    using PreparedDynamics = typename Dynamics::Prepared;
    using ScheduleData = PreparedTerminalPathSchedule<
        PathScheduleTraits<Schedule>::kIntervalCapacity
    >;

    MixedNodeGraphWorkspace<NodePolicy> graph{};
    PreparedDynamics* prepared_dynamics = nullptr;
    std::size_t prepared_dynamics_capacity = 0U;
    ScheduleData* schedules = nullptr;
    std::size_t schedule_capacity = 0U;
    std::uint16_t* node_counts = nullptr;
    std::size_t node_count_capacity = 0U;
    std::uint8_t* simulation_flags = nullptr;
    std::size_t simulation_flag_capacity = 0U;
    PreparedDynamics* terminal_dynamics = nullptr;
    std::size_t terminal_dynamics_capacity = 0U;
    std::uint32_t* terminal_transition_counts = nullptr;
    std::size_t terminal_transition_count_capacity = 0U;
    std::uint32_t* maximum_terminal_transitions = nullptr;
    std::size_t maximum_terminal_transition_capacity = 0U;
    float* terminal_normal_weights = nullptr;
    std::size_t terminal_normal_weight_capacity = 0U;
};

struct MixedPathNodeGraphWorkspaceRequirements {
    MixedNodeGraphWorkspaceRequirements graph{};
    std::size_t prepared_dynamics = 0U;
    std::size_t schedules = 0U;
    std::size_t node_counts = 0U;
    std::size_t simulation_flags = 0U;
    std::size_t terminal_dynamics = 0U;
    std::size_t terminal_transition_counts = 0U;
    std::size_t maximum_terminal_transitions = 0U;
    std::size_t terminal_normal_weights = 0U;
};

struct MixedPathNodeGraphWorkspaceLayout {
    MixedNodeGraphWorkspaceLayout graph{};
    MixedPathNodeGraphWorkspaceRequirements capacities{};
    std::size_t prepared_dynamics_offset = 0U;
    std::size_t schedules_offset = 0U;
    std::size_t node_counts_offset = 0U;
    std::size_t simulation_flags_offset = 0U;
    std::size_t terminal_dynamics_offset = 0U;
    std::size_t terminal_transition_counts_offset = 0U;
    std::size_t maximum_terminal_transitions_offset = 0U;
    std::size_t terminal_normal_weights_offset = 0U;
    std::size_t bytes = 0U;
};

template<typename NodePolicy, typename Dynamics, typename Schedule>
MixedPathNodeGraphWorkspaceLayout mixed_path_node_graph_workspace_layout(
    std::size_t sensitivity_count,
    const pg::SensitivityGraphPlan& graph_plan,
    unsigned int reduction_threads,
    TerminalNodeGraphConfiguration configuration,
    bool variable_terminal = false
) {
    const auto graph_layout = mixed_node_graph_workspace_layout<NodePolicy>(
        sensitivity_count,
        graph_plan,
        reduction_threads,
        configuration
    );
    const auto rows = configuration.row_chunk_size;
    const auto row_nodes = checked_workspace_product(
        rows,
        graph_plan.node_capacity,
        "Mixed path row-node workspace size overflow."
    );
    MixedPathNodeGraphWorkspaceLayout layout{};
    layout.graph = graph_layout;
    layout.capacities.graph = graph_layout.capacities;
    layout.capacities.prepared_dynamics = checked_workspace_product(
        row_nodes,
        path_dynamics_interval_capacity_v<Schedule>,
        "Mixed path dynamics workspace size overflow."
    );
    layout.capacities.schedules = rows;
    layout.capacities.node_counts = rows;
    layout.capacities.simulation_flags = row_nodes;
    if (variable_terminal) {
        layout.capacities.terminal_dynamics = row_nodes;
        layout.capacities.terminal_transition_counts = row_nodes;
        layout.capacities.maximum_terminal_transitions = rows;
        layout.capacities.terminal_normal_weights = checked_workspace_product(
            row_nodes,
            4U,
            "Mixed path terminal Brownian-weight workspace size overflow."
        );
    }

    std::size_t offset = graph_layout.bytes;
    layout.prepared_dynamics_offset =
        workspace_detail::append_workspace_array<
            typename Dynamics::Prepared
        >(offset, layout.capacities.prepared_dynamics);
    layout.schedules_offset = workspace_detail::append_workspace_array<
        typename MixedPathNodeGraphWorkspace<
            NodePolicy, Dynamics, Schedule
        >::ScheduleData
    >(offset, layout.capacities.schedules);
    layout.node_counts_offset = workspace_detail::append_workspace_array<
        std::uint16_t
    >(offset, layout.capacities.node_counts);
    layout.simulation_flags_offset = workspace_detail::append_workspace_array<
        std::uint8_t
    >(offset, layout.capacities.simulation_flags);
    layout.terminal_dynamics_offset =
        workspace_detail::append_workspace_array<
            typename Dynamics::Prepared
        >(offset, layout.capacities.terminal_dynamics);
    layout.terminal_transition_counts_offset =
        workspace_detail::append_workspace_array<std::uint32_t>(
            offset, layout.capacities.terminal_transition_counts
        );
    layout.maximum_terminal_transitions_offset =
        workspace_detail::append_workspace_array<std::uint32_t>(
            offset, layout.capacities.maximum_terminal_transitions
        );
    layout.terminal_normal_weights_offset =
        workspace_detail::append_workspace_array<float>(
            offset, layout.capacities.terminal_normal_weights
        );
    layout.bytes = offset;
    return layout;
}

template<typename NodePolicy, typename Dynamics, typename Schedule>
MixedPathNodeGraphWorkspace<NodePolicy, Dynamics, Schedule>
make_mixed_path_node_graph_workspace(
    void* storage,
    std::size_t storage_bytes,
    const MixedPathNodeGraphWorkspaceLayout& layout
) {
    if (storage == nullptr || storage_bytes < layout.bytes) {
        throw std::invalid_argument(
            "Insufficient mixed path node-graph byte workspace."
        );
    }
    using Workspace = MixedPathNodeGraphWorkspace<
        NodePolicy, Dynamics, Schedule
    >;
    auto* base = static_cast<unsigned char*>(storage);
    return {
        make_mixed_node_graph_workspace<NodePolicy>(
            storage, storage_bytes, layout.graph
        ),
        reinterpret_cast<typename Workspace::PreparedDynamics*>(
            base + layout.prepared_dynamics_offset
        ),
        layout.capacities.prepared_dynamics,
        reinterpret_cast<typename Workspace::ScheduleData*>(
            base + layout.schedules_offset
        ),
        layout.capacities.schedules,
        reinterpret_cast<std::uint16_t*>(base + layout.node_counts_offset),
        layout.capacities.node_counts,
        reinterpret_cast<std::uint8_t*>(
            base + layout.simulation_flags_offset
        ),
        layout.capacities.simulation_flags,
        reinterpret_cast<typename Workspace::PreparedDynamics*>(
            base + layout.terminal_dynamics_offset
        ),
        layout.capacities.terminal_dynamics,
        reinterpret_cast<std::uint32_t*>(
            base + layout.terminal_transition_counts_offset
        ),
        layout.capacities.terminal_transition_counts,
        reinterpret_cast<std::uint32_t*>(
            base + layout.maximum_terminal_transitions_offset
        ),
        layout.capacities.maximum_terminal_transitions,
        reinterpret_cast<float*>(
            base + layout.terminal_normal_weights_offset
        ),
        layout.capacities.terminal_normal_weights,
    };
}

template<typename NodePolicy, typename Dynamics, typename Schedule>
void validate_mixed_path_node_graph_workspace(
    MixedPathNodeGraphWorkspace<NodePolicy, Dynamics, Schedule> workspace,
    const MixedPathNodeGraphWorkspaceRequirements& required
) {
    validate_mixed_node_graph_workspace(workspace.graph, required.graph);
    const auto validate = [](const void* pointer,
                             std::size_t capacity,
                             std::size_t needed,
                             const char* name) {
        if (capacity < needed) {
            throw std::invalid_argument(
                std::string("Insufficient ") + name + " capacity."
            );
        }
        if (needed > 0U) validate_device_pointer(pointer, name);
    };
    validate(
        workspace.prepared_dynamics,
        workspace.prepared_dynamics_capacity,
        required.prepared_dynamics,
        "mixed path prepared dynamics"
    );
    validate(
        workspace.schedules,
        workspace.schedule_capacity,
        required.schedules,
        "mixed path schedules"
    );
    validate(
        workspace.node_counts,
        workspace.node_count_capacity,
        required.node_counts,
        "mixed path node counts"
    );
    validate(
        workspace.simulation_flags,
        workspace.simulation_flag_capacity,
        required.simulation_flags,
        "mixed path simulation flags"
    );
    validate(
        workspace.terminal_dynamics,
        workspace.terminal_dynamics_capacity,
        required.terminal_dynamics,
        "mixed path terminal dynamics"
    );
    validate(
        workspace.terminal_transition_counts,
        workspace.terminal_transition_count_capacity,
        required.terminal_transition_counts,
        "mixed path terminal transition counts"
    );
    validate(
        workspace.maximum_terminal_transitions,
        workspace.maximum_terminal_transition_capacity,
        required.maximum_terminal_transitions,
        "mixed path maximum terminal transitions"
    );
    validate(
        workspace.terminal_normal_weights,
        workspace.terminal_normal_weight_capacity,
        required.terminal_normal_weights,
        "mixed path terminal Brownian weights"
    );
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
