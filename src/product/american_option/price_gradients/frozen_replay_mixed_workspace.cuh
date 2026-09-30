// Persistent workspaces for both mixed American frozen replays.
#pragma once

#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/value_policy.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::american_option::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
namespace lspg =
    ::ai_factory::workbench::longstaff_schwartz::price_gradients;

namespace mixed_workspace_detail {

template<typename Replay>
inline constexpr std::size_t interval_capacity_v =
    Replay::kExactTransitionReplay ? 2U : 1U;

}  // namespace mixed_workspace_detail

template<typename Replay>
struct FrozenExerciseMixedWorkspace {
    using PreparedDynamics = typename Replay::Dynamics::Prepared;

    mcpg::MixedNodeGraphWorkspace<
        lspg::FrozenReplayValueNodePolicy
    > graph{};
    PreparedDynamics* prepared_dynamics = nullptr;
    std::size_t prepared_dynamics_capacity = 0U;
    float* spot_scales = nullptr;
    float* strikes = nullptr;
    float* initial_spots = nullptr;
    float* initial_discounts = nullptr;
    float* exercise_discounts = nullptr;
    std::size_t node_data_capacity = 0U;
    std::uint8_t* reuse_central = nullptr;
    std::size_t reuse_central_capacity = 0U;
    std::uint16_t* node_counts = nullptr;
    std::size_t node_count_capacity = 0U;
};

struct FrozenExerciseMixedWorkspaceRequirements {
    mcpg::MixedNodeGraphWorkspaceRequirements graph{};
    std::size_t prepared_dynamics = 0U;
    std::size_t node_data = 0U;
    std::size_t reuse_central = 0U;
    std::size_t node_counts = 0U;
};

struct FrozenExerciseMixedWorkspaceLayout {
    mcpg::MixedNodeGraphWorkspaceLayout graph{};
    FrozenExerciseMixedWorkspaceRequirements capacities{};
    std::size_t prepared_dynamics_offset = 0U;
    std::size_t spot_scales_offset = 0U;
    std::size_t strikes_offset = 0U;
    std::size_t initial_spots_offset = 0U;
    std::size_t initial_discounts_offset = 0U;
    std::size_t exercise_discounts_offset = 0U;
    std::size_t reuse_central_offset = 0U;
    std::size_t node_counts_offset = 0U;
    std::size_t bytes = 0U;
};

template<typename Replay>
FrozenExerciseMixedWorkspaceLayout frozen_exercise_mixed_workspace_layout(
    std::size_t sensitivity_count,
    const pg::SensitivityGraphPlan& graph_plan,
    unsigned int reduction_threads,
    mcpg::TerminalNodeGraphConfiguration configuration,
    std::size_t prepared_node_capacity
) {
    const auto graph_layout =
        mcpg::mixed_node_graph_workspace_layout<
            lspg::FrozenReplayValueNodePolicy
        >(
            sensitivity_count,
            graph_plan,
            reduction_threads,
            configuration
        );
    const auto row_nodes = checked_workspace_product(
        configuration.row_chunk_size,
        graph_plan.node_capacity,
        "American mixed replay row-node workspace size overflow."
    );

    FrozenExerciseMixedWorkspaceLayout layout{};
    layout.graph = graph_layout;
    layout.capacities.graph = graph_layout.capacities;
    const auto prepared_row_nodes = checked_workspace_product(
        configuration.row_chunk_size,
        prepared_node_capacity,
        "American mixed replay prepared-node workspace size overflow."
    );
    layout.capacities.prepared_dynamics = checked_workspace_product(
        prepared_row_nodes,
        mixed_workspace_detail::interval_capacity_v<Replay>,
        "American mixed replay dynamics workspace size overflow."
    );
    layout.capacities.node_data = row_nodes;
    layout.capacities.reuse_central = row_nodes;
    layout.capacities.node_counts = configuration.row_chunk_size;

    std::size_t offset = graph_layout.bytes;
    layout.prepared_dynamics_offset =
        mcpg::workspace_detail::append_workspace_array<
            typename Replay::Dynamics::Prepared
        >(offset, layout.capacities.prepared_dynamics);
    layout.spot_scales_offset =
        mcpg::workspace_detail::append_workspace_array<float>(
            offset, layout.capacities.node_data
        );
    layout.strikes_offset =
        mcpg::workspace_detail::append_workspace_array<float>(
            offset, layout.capacities.node_data
        );
    layout.initial_spots_offset =
        mcpg::workspace_detail::append_workspace_array<float>(
            offset, layout.capacities.node_data
        );
    layout.initial_discounts_offset =
        mcpg::workspace_detail::append_workspace_array<float>(
            offset, layout.capacities.node_data
        );
    layout.exercise_discounts_offset =
        mcpg::workspace_detail::append_workspace_array<float>(
            offset, layout.capacities.node_data
        );
    layout.reuse_central_offset =
        mcpg::workspace_detail::append_workspace_array<std::uint8_t>(
            offset, layout.capacities.reuse_central
        );
    layout.node_counts_offset =
        mcpg::workspace_detail::append_workspace_array<std::uint16_t>(
            offset, layout.capacities.node_counts
        );
    layout.bytes = offset;
    return layout;
}

template<typename Replay>
FrozenExerciseMixedWorkspace<Replay> make_frozen_exercise_mixed_workspace(
    void* storage,
    std::size_t storage_bytes,
    const FrozenExerciseMixedWorkspaceLayout& layout
) {
    if (storage == nullptr || storage_bytes < layout.bytes) {
        throw std::invalid_argument(
            "Insufficient American mixed frozen-replay byte workspace."
        );
    }
    using Workspace = FrozenExerciseMixedWorkspace<Replay>;
    auto* base = static_cast<unsigned char*>(storage);
    return {
        mcpg::make_mixed_node_graph_workspace<
            lspg::FrozenReplayValueNodePolicy
        >(storage, storage_bytes, layout.graph),
        reinterpret_cast<typename Workspace::PreparedDynamics*>(
            base + layout.prepared_dynamics_offset
        ),
        layout.capacities.prepared_dynamics,
        reinterpret_cast<float*>(base + layout.spot_scales_offset),
        reinterpret_cast<float*>(base + layout.strikes_offset),
        reinterpret_cast<float*>(base + layout.initial_spots_offset),
        reinterpret_cast<float*>(base + layout.initial_discounts_offset),
        reinterpret_cast<float*>(base + layout.exercise_discounts_offset),
        layout.capacities.node_data,
        reinterpret_cast<std::uint8_t*>(
            base + layout.reuse_central_offset
        ),
        layout.capacities.reuse_central,
        reinterpret_cast<std::uint16_t*>(base + layout.node_counts_offset),
        layout.capacities.node_counts,
    };
}

template<typename Replay>
void validate_frozen_exercise_mixed_workspace(
    FrozenExerciseMixedWorkspace<Replay> workspace,
    const FrozenExerciseMixedWorkspaceRequirements& required
) {
    mcpg::validate_mixed_node_graph_workspace(
        workspace.graph, required.graph
    );
    const auto validate = [](const void* pointer,
                             std::size_t capacity,
                             std::size_t needed,
                             const char* name) {
        if (capacity < needed) {
            throw std::invalid_argument(
                std::string("Insufficient ") + name + " capacity."
            );
        }
        validate_device_pointer(pointer, name);
    };
    validate(
        workspace.prepared_dynamics,
        workspace.prepared_dynamics_capacity,
        required.prepared_dynamics,
        "American mixed replay prepared dynamics"
    );
    validate(
        workspace.spot_scales,
        workspace.node_data_capacity,
        required.node_data,
        "American mixed replay node data"
    );
    validate(
        workspace.strikes,
        workspace.node_data_capacity,
        required.node_data,
        "American mixed replay strikes"
    );
    validate(
        workspace.initial_spots,
        workspace.node_data_capacity,
        required.node_data,
        "American mixed replay initial spots"
    );
    validate(
        workspace.initial_discounts,
        workspace.node_data_capacity,
        required.node_data,
        "American mixed replay initial discounts"
    );
    validate(
        workspace.exercise_discounts,
        workspace.node_data_capacity,
        required.node_data,
        "American mixed replay exercise discounts"
    );
    validate(
        workspace.reuse_central,
        workspace.reuse_central_capacity,
        required.reuse_central,
        "American mixed replay central-reuse flags"
    );
    validate(
        workspace.node_counts,
        workspace.node_count_capacity,
        required.node_counts,
        "American mixed replay node counts"
    );
}

// Frozen-policy replay stores complete prepared node rows. Unlike the
// frozen-exercise layout, it does not retain coupled interval dynamics: every
// node must inspect every exercise date against the captured central
// continuation functions.
template<typename CentralPolicy>
struct FrozenRegressionMixedWorkspace {
    using PreparedRow = typename CentralPolicy::PreparedRow;

    mcpg::MixedNodeGraphWorkspace<
        lspg::FrozenReplayValueNodePolicy
    > graph{};
    PreparedRow* prepared_nodes = nullptr;
    std::size_t prepared_node_capacity = 0U;
    std::uint16_t* node_counts = nullptr;
    std::size_t node_count_capacity = 0U;
};

struct FrozenRegressionMixedWorkspaceRequirements {
    mcpg::MixedNodeGraphWorkspaceRequirements graph{};
    std::size_t prepared_nodes = 0U;
    std::size_t node_counts = 0U;
};

struct FrozenRegressionMixedWorkspaceLayout {
    mcpg::MixedNodeGraphWorkspaceLayout graph{};
    FrozenRegressionMixedWorkspaceRequirements capacities{};
    std::size_t prepared_nodes_offset = 0U;
    std::size_t node_counts_offset = 0U;
    std::size_t bytes = 0U;
};

template<typename CentralPolicy>
FrozenRegressionMixedWorkspaceLayout
frozen_regression_mixed_workspace_layout(
    std::size_t sensitivity_count,
    const pg::SensitivityGraphPlan& graph_plan,
    unsigned int reduction_threads,
    mcpg::TerminalNodeGraphConfiguration configuration
) {
    const auto graph_layout =
        mcpg::mixed_node_graph_workspace_layout<
            lspg::FrozenReplayValueNodePolicy
        >(
            sensitivity_count,
            graph_plan,
            reduction_threads,
            configuration
        );
    const auto row_nodes = checked_workspace_product(
        configuration.row_chunk_size,
        graph_plan.node_capacity,
        "American frozen-policy prepared-node workspace size overflow."
    );

    FrozenRegressionMixedWorkspaceLayout layout{};
    layout.graph = graph_layout;
    layout.capacities.graph = graph_layout.capacities;
    layout.capacities.prepared_nodes = row_nodes;
    layout.capacities.node_counts = configuration.row_chunk_size;

    std::size_t offset = graph_layout.bytes;
    layout.prepared_nodes_offset =
        mcpg::workspace_detail::append_workspace_array<
            typename CentralPolicy::PreparedRow
        >(offset, layout.capacities.prepared_nodes);
    layout.node_counts_offset =
        mcpg::workspace_detail::append_workspace_array<std::uint16_t>(
            offset, layout.capacities.node_counts
        );
    layout.bytes = offset;
    return layout;
}

template<typename CentralPolicy>
FrozenRegressionMixedWorkspace<CentralPolicy>
make_frozen_regression_mixed_workspace(
    void* storage,
    std::size_t storage_bytes,
    const FrozenRegressionMixedWorkspaceLayout& layout
) {
    if (storage == nullptr || storage_bytes < layout.bytes) {
        throw std::invalid_argument(
            "Insufficient American mixed frozen-policy byte workspace."
        );
    }
    using Workspace = FrozenRegressionMixedWorkspace<CentralPolicy>;
    auto* base = static_cast<unsigned char*>(storage);
    return {
        mcpg::make_mixed_node_graph_workspace<
            lspg::FrozenReplayValueNodePolicy
        >(storage, storage_bytes, layout.graph),
        reinterpret_cast<typename Workspace::PreparedRow*>(
            base + layout.prepared_nodes_offset
        ),
        layout.capacities.prepared_nodes,
        reinterpret_cast<std::uint16_t*>(base + layout.node_counts_offset),
        layout.capacities.node_counts,
    };
}

template<typename CentralPolicy>
void validate_frozen_regression_mixed_workspace(
    FrozenRegressionMixedWorkspace<CentralPolicy> workspace,
    const FrozenRegressionMixedWorkspaceRequirements& required
) {
    mcpg::validate_mixed_node_graph_workspace(
        workspace.graph, required.graph
    );
    if (workspace.prepared_node_capacity < required.prepared_nodes
        || workspace.node_count_capacity < required.node_counts) {
        throw std::invalid_argument(
            "Insufficient American mixed frozen-policy workspace capacity."
        );
    }
    validate_device_pointer(
        workspace.prepared_nodes,
        "American mixed frozen-policy prepared nodes"
    );
    validate_device_pointer(
        workspace.node_counts,
        "American mixed frozen-policy node counts"
    );
}

}  // namespace ai_factory::workbench::product::american_option::price_gradients
