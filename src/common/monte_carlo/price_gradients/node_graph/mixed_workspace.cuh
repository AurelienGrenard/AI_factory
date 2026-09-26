// Engine-neutral workspace for selected mixed sensitivity node graphs.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/node_indices.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/price_gradients/device_sensitivity_graph.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil_outputs.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

template<typename NodePolicy>
struct MixedNodeGraphWorkspace {
    using NodeValue = typename NodePolicy::NodeValue;
    using NodeMetadata = typename NodePolicy::Metadata;
    using AxisNodeIndices = SensitivityNodeIndices<4U>;
    using MixedNodeIndices = MixedSensitivityNodeIndices;

    std::uint16_t* first_coordinates = nullptr;
    std::size_t first_coordinate_capacity = 0U;
    std::uint16_t* diagonal_coordinates = nullptr;
    std::size_t diagonal_coordinate_capacity = 0U;
    pg::SensitivityPair* mixed_pairs = nullptr;
    std::size_t mixed_pair_capacity = 0U;
    pg::SensitivityCoordinateUse* coordinate_uses = nullptr;
    std::size_t coordinate_use_capacity = 0U;
    NodeValue* node_values = nullptr;
    std::size_t node_value_capacity = 0U;
    NodeMetadata* node_metadata = nullptr;
    std::size_t node_metadata_capacity = 0U;
    AxisNodeIndices* axis_node_indices = nullptr;
    std::size_t axis_node_index_capacity = 0U;
    MixedNodeIndices* mixed_node_indices = nullptr;
    std::size_t mixed_node_index_capacity = 0U;
    std::uint8_t* row_status = nullptr;
    std::size_t row_status_capacity = 0U;
    reductions::MomentSums* thread_moments = nullptr;
    std::size_t thread_moment_capacity = 0U;
};

struct MixedNodeGraphWorkspaceRequirements {
    std::size_t first_coordinates = 0U;
    std::size_t diagonal_coordinates = 0U;
    std::size_t mixed_pairs = 0U;
    std::size_t coordinate_uses = 0U;
    std::size_t node_values = 0U;
    std::size_t node_metadata = 0U;
    std::size_t axis_node_indices = 0U;
    std::size_t mixed_node_indices = 0U;
    std::size_t row_status = 0U;
    std::size_t thread_moments = 0U;
};

struct MixedNodeGraphWorkspaceLayout {
    MixedNodeGraphWorkspaceRequirements capacities{};
    std::size_t first_coordinates_offset = 0U;
    std::size_t diagonal_coordinates_offset = 0U;
    std::size_t mixed_pairs_offset = 0U;
    std::size_t coordinate_uses_offset = 0U;
    std::size_t node_values_offset = 0U;
    std::size_t node_metadata_offset = 0U;
    std::size_t axis_node_indices_offset = 0U;
    std::size_t mixed_node_indices_offset = 0U;
    std::size_t row_status_offset = 0U;
    std::size_t thread_moments_offset = 0U;
    std::size_t bytes = 0U;
};

inline MixedNodeGraphWorkspaceRequirements
mixed_node_graph_workspace_requirements(
    std::size_t sensitivity_count,
    const pg::SensitivityGraphPlan& graph_plan,
    unsigned int reduction_threads,
    TerminalNodeGraphConfiguration configuration
) {
    if (sensitivity_count == 0U
        || graph_plan.coordinate_uses.size() != sensitivity_count
        || graph_plan.mixed_second.empty()
        || graph_plan.node_capacity == 0U) {
        throw std::invalid_argument(
            "The mixed node-graph sensitivity plan is invalid."
        );
    }
    if (configuration.row_chunk_size == 0U
        || configuration.path_chunk_size == 0U
        || configuration.path_shards == 0U) {
        throw std::invalid_argument(
            "Mixed node-graph chunk sizes and path shards must be positive."
        );
    }
    if (reduction_threads < 32U || reduction_threads > 1024U
        || reduction_threads % 32U != 0U
        || configuration.path_chunk_size % reduction_threads != 0U) {
        throw std::invalid_argument(
            "Mixed node-graph reduction geometry is invalid."
        );
    }

    const auto rows = configuration.row_chunk_size;
    return {
        graph_plan.first.size(),
        graph_plan.diagonal_second.size(),
        graph_plan.mixed_second.size(),
        graph_plan.coordinate_uses.size(),
        checked_workspace_product(
            checked_workspace_product(
                rows,
                configuration.path_chunk_size,
                "Mixed node-value workspace size overflow."
            ),
            graph_plan.node_capacity,
            "Mixed node-value workspace size overflow."
        ),
        checked_workspace_product(
            rows,
            graph_plan.node_capacity,
            "Mixed node-metadata workspace size overflow."
        ),
        checked_workspace_product(
            rows,
            sensitivity_count,
            "Mixed axis-index workspace size overflow."
        ),
        checked_workspace_product(
            rows,
            graph_plan.mixed_second.size(),
            "Mixed pair-index workspace size overflow."
        ),
        rows,
        checked_workspace_product(
            checked_workspace_product(
                rows,
                graph_plan.output_count(),
                "Mixed moment workspace size overflow."
            ),
            reduction_threads,
            "Mixed moment workspace size overflow."
        ),
    };
}

template<typename NodePolicy>
MixedNodeGraphWorkspaceLayout mixed_node_graph_workspace_layout(
    std::size_t sensitivity_count,
    const pg::SensitivityGraphPlan& graph_plan,
    unsigned int reduction_threads,
    TerminalNodeGraphConfiguration configuration
) {
    const auto capacities = mixed_node_graph_workspace_requirements(
        sensitivity_count, graph_plan, reduction_threads, configuration
    );
    using Workspace = MixedNodeGraphWorkspace<NodePolicy>;
    std::size_t offset = 0U;
    MixedNodeGraphWorkspaceLayout layout{};
    layout.capacities = capacities;
    layout.first_coordinates_offset = workspace_detail::append_workspace_array<
        std::uint16_t
    >(offset, capacities.first_coordinates);
    layout.diagonal_coordinates_offset = workspace_detail::append_workspace_array<
        std::uint16_t
    >(offset, capacities.diagonal_coordinates);
    layout.mixed_pairs_offset = workspace_detail::append_workspace_array<
        pg::SensitivityPair
    >(offset, capacities.mixed_pairs);
    layout.coordinate_uses_offset = workspace_detail::append_workspace_array<
        pg::SensitivityCoordinateUse
    >(offset, capacities.coordinate_uses);
    layout.node_values_offset = workspace_detail::append_workspace_array<
        typename Workspace::NodeValue
    >(offset, capacities.node_values);
    layout.node_metadata_offset = workspace_detail::append_workspace_array<
        typename Workspace::NodeMetadata
    >(offset, capacities.node_metadata);
    layout.axis_node_indices_offset = workspace_detail::append_workspace_array<
        typename Workspace::AxisNodeIndices
    >(offset, capacities.axis_node_indices);
    layout.mixed_node_indices_offset = workspace_detail::append_workspace_array<
        typename Workspace::MixedNodeIndices
    >(offset, capacities.mixed_node_indices);
    layout.row_status_offset = workspace_detail::append_workspace_array<
        std::uint8_t
    >(offset, capacities.row_status);
    layout.thread_moments_offset = workspace_detail::append_workspace_array<
        reductions::MomentSums
    >(offset, capacities.thread_moments);
    layout.bytes = offset;
    return layout;
}

template<typename NodePolicy>
MixedNodeGraphWorkspace<NodePolicy>
make_mixed_node_graph_workspace(
    void* storage,
    std::size_t storage_bytes,
    const MixedNodeGraphWorkspaceLayout& layout
) {
    if (storage == nullptr || storage_bytes < layout.bytes) {
        throw std::invalid_argument(
            "Insufficient mixed node-graph byte workspace."
        );
    }
    using Workspace = MixedNodeGraphWorkspace<NodePolicy>;
    auto* base = static_cast<unsigned char*>(storage);
    return {
        reinterpret_cast<std::uint16_t*>(
            base + layout.first_coordinates_offset
        ),
        layout.capacities.first_coordinates,
        reinterpret_cast<std::uint16_t*>(
            base + layout.diagonal_coordinates_offset
        ),
        layout.capacities.diagonal_coordinates,
        reinterpret_cast<pg::SensitivityPair*>(
            base + layout.mixed_pairs_offset
        ),
        layout.capacities.mixed_pairs,
        reinterpret_cast<pg::SensitivityCoordinateUse*>(
            base + layout.coordinate_uses_offset
        ),
        layout.capacities.coordinate_uses,
        reinterpret_cast<typename Workspace::NodeValue*>(
            base + layout.node_values_offset
        ),
        layout.capacities.node_values,
        reinterpret_cast<typename Workspace::NodeMetadata*>(
            base + layout.node_metadata_offset
        ),
        layout.capacities.node_metadata,
        reinterpret_cast<typename Workspace::AxisNodeIndices*>(
            base + layout.axis_node_indices_offset
        ),
        layout.capacities.axis_node_indices,
        reinterpret_cast<typename Workspace::MixedNodeIndices*>(
            base + layout.mixed_node_indices_offset
        ),
        layout.capacities.mixed_node_indices,
        reinterpret_cast<std::uint8_t*>(base + layout.row_status_offset),
        layout.capacities.row_status,
        reinterpret_cast<reductions::MomentSums*>(
            base + layout.thread_moments_offset
        ),
        layout.capacities.thread_moments,
    };
}

template<typename NodePolicy>
void validate_mixed_node_graph_workspace(
    MixedNodeGraphWorkspace<NodePolicy> workspace,
    MixedNodeGraphWorkspaceRequirements required
) {
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
    validate(workspace.first_coordinates,
             workspace.first_coordinate_capacity,
             required.first_coordinates,
             "mixed graph first coordinates");
    validate(workspace.diagonal_coordinates,
             workspace.diagonal_coordinate_capacity,
             required.diagonal_coordinates,
             "mixed graph diagonal coordinates");
    validate(workspace.mixed_pairs,
             workspace.mixed_pair_capacity,
             required.mixed_pairs,
             "mixed graph coordinate pairs");
    validate(workspace.coordinate_uses,
             workspace.coordinate_use_capacity,
             required.coordinate_uses,
             "mixed graph coordinate uses");
    validate(workspace.node_values, workspace.node_value_capacity,
             required.node_values, "mixed node values");
    validate(workspace.node_metadata, workspace.node_metadata_capacity,
             required.node_metadata, "mixed node metadata");
    validate(workspace.axis_node_indices, workspace.axis_node_index_capacity,
             required.axis_node_indices, "mixed axis indices");
    validate(workspace.mixed_node_indices, workspace.mixed_node_index_capacity,
             required.mixed_node_indices, "mixed pair indices");
    validate(workspace.row_status, workspace.row_status_capacity,
             required.row_status, "mixed row status");
    validate(workspace.thread_moments, workspace.thread_moment_capacity,
             required.thread_moments, "mixed thread moments");
}


template<typename NodePolicy>
pg::DeviceSensitivityGraph upload_mixed_sensitivity_graph(
    MixedNodeGraphWorkspace<NodePolicy> workspace,
    const pg::SensitivityGraphPlan& graph
) {
    return pg::upload_sensitivity_graph(workspace, graph);
}


}  // namespace ai_factory::workbench::monte_carlo::price_gradients
