// Persistent row preparation for mixed Bermudan frozen-exercise replay.
#pragma once

#include "common/longstaff_schwartz/price_gradients/frozen_exercise_node_graph/value_policy.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::bermudan_swaption::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
namespace lspg =
    ::ai_factory::workbench::longstaff_schwartz::price_gradients;

namespace mixed_workspace_detail {

template<typename CentralPolicy, bool TerminalForward>
struct ObservationType {
    using type = std::uint8_t;
};

template<typename CentralPolicy>
struct ObservationType<CentralPolicy, true> {
    using type = typename CentralPolicy::Schedule::Observation;
};

template<typename CentralPolicy, bool TerminalForward>
using observation_t =
    typename ObservationType<CentralPolicy, TerminalForward>::type;

}  // namespace mixed_workspace_detail

template<typename CentralPolicy, bool TerminalForward>
struct FrozenExerciseMixedWorkspace {
    using PreparedRow = typename CentralPolicy::PreparedRow;
    using Observation = mixed_workspace_detail::observation_t<
        CentralPolicy, TerminalForward
    >;

    mcpg::MixedNodeGraphWorkspace<
        lspg::FrozenExerciseValueNodePolicy
    > graph{};
    PreparedRow* prepared_nodes = nullptr;
    std::size_t prepared_node_capacity = 0U;
    Observation* observations = nullptr;
    std::size_t observation_capacity = 0U;
    std::size_t observation_stride = 0U;
    std::uint16_t* node_counts = nullptr;
    std::size_t node_count_capacity = 0U;
};

struct FrozenExerciseMixedWorkspaceRequirements {
    mcpg::MixedNodeGraphWorkspaceRequirements graph{};
    std::size_t prepared_nodes = 0U;
    std::size_t observations = 0U;
    std::size_t node_counts = 0U;
};

struct FrozenExerciseMixedWorkspaceLayout {
    mcpg::MixedNodeGraphWorkspaceLayout graph{};
    FrozenExerciseMixedWorkspaceRequirements capacities{};
    std::size_t prepared_nodes_offset = 0U;
    std::size_t observations_offset = 0U;
    std::size_t observation_stride = 0U;
    std::size_t node_counts_offset = 0U;
    std::size_t bytes = 0U;
};

template<typename CentralPolicy, bool TerminalForward>
FrozenExerciseMixedWorkspaceLayout frozen_exercise_mixed_workspace_layout(
    std::size_t sensitivity_count,
    const pg::SensitivityGraphPlan& graph_plan,
    unsigned int reduction_threads,
    mcpg::TerminalNodeGraphConfiguration configuration,
    std::size_t maximum_exercise_count
) {
    const auto graph_layout =
        mcpg::mixed_node_graph_workspace_layout<
            lspg::FrozenExerciseValueNodePolicy
        >(
            sensitivity_count,
            graph_plan,
            reduction_threads,
            configuration
        );
    const auto row_nodes = checked_workspace_product(
        configuration.row_chunk_size,
        graph_plan.node_capacity,
        "Bermudan mixed replay row-node workspace size overflow."
    );

    FrozenExerciseMixedWorkspaceLayout layout{};
    layout.graph = graph_layout;
    layout.capacities.graph = graph_layout.capacities;
    layout.capacities.prepared_nodes = row_nodes;
    layout.capacities.observations = TerminalForward
        ? checked_workspace_product(
            row_nodes,
            maximum_exercise_count,
            "Bermudan mixed replay observation workspace size overflow."
        )
        : 0U;
    layout.capacities.node_counts = configuration.row_chunk_size;
    layout.observation_stride = TerminalForward
        ? maximum_exercise_count
        : 0U;

    std::size_t offset = graph_layout.bytes;
    layout.prepared_nodes_offset =
        mcpg::workspace_detail::append_workspace_array<
            typename CentralPolicy::PreparedRow
        >(offset, layout.capacities.prepared_nodes);
    layout.observations_offset =
        mcpg::workspace_detail::append_workspace_array<
            mixed_workspace_detail::observation_t<
            CentralPolicy, TerminalForward
        >>(offset, layout.capacities.observations);
    layout.node_counts_offset =
        mcpg::workspace_detail::append_workspace_array<std::uint16_t>(
            offset, layout.capacities.node_counts
        );
    layout.bytes = offset;
    return layout;
}

template<typename CentralPolicy, bool TerminalForward>
FrozenExerciseMixedWorkspace<CentralPolicy, TerminalForward>
make_frozen_exercise_mixed_workspace(
    void* storage,
    std::size_t storage_bytes,
    const FrozenExerciseMixedWorkspaceLayout& layout
) {
    if (storage == nullptr || storage_bytes < layout.bytes) {
        throw std::invalid_argument(
            "Insufficient Bermudan mixed frozen-replay byte workspace."
        );
    }
    using Workspace = FrozenExerciseMixedWorkspace<
        CentralPolicy, TerminalForward
    >;
    auto* base = static_cast<unsigned char*>(storage);
    return {
        mcpg::make_mixed_node_graph_workspace<
            lspg::FrozenExerciseValueNodePolicy
        >(storage, storage_bytes, layout.graph),
        reinterpret_cast<typename Workspace::PreparedRow*>(
            base + layout.prepared_nodes_offset
        ),
        layout.capacities.prepared_nodes,
        reinterpret_cast<typename Workspace::Observation*>(
            base + layout.observations_offset
        ),
        layout.capacities.observations,
        layout.observation_stride,
        reinterpret_cast<std::uint16_t*>(base + layout.node_counts_offset),
        layout.capacities.node_counts,
    };
}

template<typename CentralPolicy, bool TerminalForward>
void validate_frozen_exercise_mixed_workspace(
    FrozenExerciseMixedWorkspace<CentralPolicy, TerminalForward> workspace,
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
        if (needed != 0U) validate_device_pointer(pointer, name);
    };
    validate(
        workspace.prepared_nodes,
        workspace.prepared_node_capacity,
        required.prepared_nodes,
        "Bermudan mixed replay prepared nodes"
    );
    validate(
        workspace.observations,
        workspace.observation_capacity,
        required.observations,
        "Bermudan mixed replay observations"
    );
    validate(
        workspace.node_counts,
        workspace.node_count_capacity,
        required.node_counts,
        "Bermudan mixed replay node counts"
    );
}

}  // namespace ai_factory::workbench::product::bermudan_swaption::price_gradients
