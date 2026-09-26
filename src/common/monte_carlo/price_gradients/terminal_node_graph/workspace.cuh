// Caller-owned bounded workspace for terminal sensitivity node graphs.
#pragma once

#include "common/check_cuda.cuh"
#include "common/price_gradients/sensitivity_request.hpp"
#include "common/reductions.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<std::size_t NodeCapacity>
struct SensitivityNodeIndices {
    std::uint16_t values[NodeCapacity]{};

    __host__ __device__ std::uint16_t& operator[](std::size_t index) {
        return values[index];
    }

    __host__ __device__ std::uint16_t operator[](
        std::size_t index
    ) const {
        return values[index];
    }
};

struct TerminalNodeGraphConfiguration {
    std::size_t row_chunk_size = 1U;
    std::size_t path_chunk_size = 1U << 20U;
    unsigned int path_shards = 1U;
};

template<typename NodePolicy>
struct TerminalNodeGraphWorkspace {
    using NodeValue = typename NodePolicy::NodeValue;
    using NodeMetadata = typename NodePolicy::Metadata;
    using NodeIndices = SensitivityNodeIndices<4U>;

    NodeValue* node_values = nullptr;
    std::size_t node_value_capacity = 0U;
    NodeMetadata* node_metadata = nullptr;
    std::size_t node_metadata_capacity = 0U;
    NodeIndices* node_indices = nullptr;
    std::size_t node_index_capacity = 0U;
    std::uint8_t* row_status = nullptr;
    std::size_t row_status_capacity = 0U;
    reductions::MomentSums* thread_moments = nullptr;
    std::size_t thread_moment_capacity = 0U;
};

struct TerminalNodeGraphWorkspaceRequirements {
    std::size_t node_values = 0U;
    std::size_t node_metadata = 0U;
    std::size_t node_indices = 0U;
    std::size_t row_status = 0U;
    std::size_t thread_moments = 0U;
};

struct TerminalNodeGraphWorkspaceLayout {
    TerminalNodeGraphWorkspaceRequirements capacities{};
    std::size_t node_values_offset = 0U;
    std::size_t node_metadata_offset = 0U;
    std::size_t node_indices_offset = 0U;
    std::size_t row_status_offset = 0U;
    std::size_t thread_moments_offset = 0U;
    std::size_t bytes = 0U;
};

template<pg::SensitivityOrders Orders>
__host__ __device__ constexpr std::size_t terminal_node_graph_output_count(
    std::size_t sensitivity_count
) {
    static_assert(pg::requests_second_v<Orders>);
    constexpr std::size_t channels =
        (pg::requests_first_v<Orders> ? 1U : 0U) + 1U;
    return 1U + channels * sensitivity_count;
}

template<std::size_t MaximumSensitivities>
__host__ __device__ constexpr std::size_t terminal_node_graph_node_capacity() {
    static_assert(
        MaximumSensitivities <= (0xffffU - 1U) / 3U,
        "Terminal node indices exceed their compact representation."
    );
    // Central plus at most three non-central nodes for every four-node
    // diagonal stencil. Centered rows naturally use only two of the slots.
    return 1U + 3U * MaximumSensitivities;
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
TerminalNodeGraphWorkspaceRequirements
terminal_node_graph_workspace_requirements(
    std::size_t sensitivity_count,
    unsigned int reduction_threads,
    TerminalNodeGraphConfiguration configuration
) {
    static_assert(pg::requests_second_v<Orders>);
    if (sensitivity_count == 0U
        || sensitivity_count > MaximumSensitivities) {
        throw std::invalid_argument(
            "The node-graph sensitivity count is unsupported."
        );
    }
    if (configuration.row_chunk_size == 0U
        || configuration.path_chunk_size == 0U
        || configuration.path_shards == 0U) {
        throw std::invalid_argument(
            "Node-graph chunk sizes and path shards must be positive."
        );
    }
    if (reduction_threads < 32U
        || reduction_threads > 1024U
        || reduction_threads % 32U != 0U) {
        throw std::invalid_argument(
            "Node-graph reduction threads must contain whole warps."
        );
    }
    if (configuration.path_chunk_size % reduction_threads != 0U) {
        throw std::invalid_argument(
            "Node-graph path chunks must be a multiple of reduction threads."
        );
    }

    constexpr auto nodes =
        terminal_node_graph_node_capacity<MaximumSensitivities>();
    const auto rows = configuration.row_chunk_size;
    const auto outputs =
        terminal_node_graph_output_count<Orders>(sensitivity_count);
    return {
        checked_workspace_product(
            checked_workspace_product(
                rows,
                configuration.path_chunk_size,
                "Node-value workspace size overflow."
            ),
            nodes,
            "Node-value workspace size overflow."
        ),
        checked_workspace_product(
            rows, nodes, "Node-metadata workspace size overflow."
        ),
        checked_workspace_product(
            rows,
            sensitivity_count,
            "Node-index workspace size overflow."
        ),
        rows,
        checked_workspace_product(
            checked_workspace_product(
                rows,
                outputs,
                "Moment workspace size overflow."
            ),
            reduction_threads,
            "Moment workspace size overflow."
        ),
    };
}

namespace workspace_detail {

inline std::size_t align_workspace_offset(
    std::size_t offset,
    std::size_t alignment
) {
    const auto remainder = offset % alignment;
    if (remainder == 0U) return offset;
    const auto padding = alignment - remainder;
    if (offset > std::numeric_limits<std::size_t>::max() - padding) {
        throw std::overflow_error("Terminal node-graph alignment overflow.");
    }
    return offset + padding;
}

template<typename Value>
std::size_t append_workspace_array(
    std::size_t& offset,
    std::size_t count
) {
    offset = align_workspace_offset(offset, alignof(Value));
    const auto result = offset;
    const auto bytes = checked_workspace_product(
        count,
        sizeof(Value),
        "Terminal node-graph byte size overflow."
    );
    if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error("Terminal node-graph byte size overflow.");
    }
    offset += bytes;
    return result;
}

}  // namespace workspace_detail

template<
    typename NodePolicy,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
TerminalNodeGraphWorkspaceLayout terminal_node_graph_workspace_layout(
    std::size_t sensitivity_count,
    unsigned int reduction_threads,
    TerminalNodeGraphConfiguration configuration
) {
    const auto capacities = terminal_node_graph_workspace_requirements<
        Orders, MaximumSensitivities
    >(sensitivity_count, reduction_threads, configuration);
    using Workspace = TerminalNodeGraphWorkspace<NodePolicy>;
    std::size_t offset = 0U;
    TerminalNodeGraphWorkspaceLayout layout{};
    layout.capacities = capacities;
    layout.node_values_offset =
        workspace_detail::append_workspace_array<
            typename Workspace::NodeValue
        >(offset, capacities.node_values);
    layout.node_metadata_offset =
        workspace_detail::append_workspace_array<
            typename Workspace::NodeMetadata
        >(offset, capacities.node_metadata);
    layout.node_indices_offset =
        workspace_detail::append_workspace_array<
            typename Workspace::NodeIndices
        >(offset, capacities.node_indices);
    layout.row_status_offset =
        workspace_detail::append_workspace_array<std::uint8_t>(
            offset, capacities.row_status
        );
    layout.thread_moments_offset =
        workspace_detail::append_workspace_array<reductions::MomentSums>(
            offset, capacities.thread_moments
        );
    layout.bytes = offset;
    return layout;
}

template<typename NodePolicy>
TerminalNodeGraphWorkspace<NodePolicy> make_terminal_node_graph_workspace(
    void* storage,
    std::size_t storage_bytes,
    const TerminalNodeGraphWorkspaceLayout& layout
) {
    if (storage == nullptr || storage_bytes < layout.bytes) {
        throw std::invalid_argument(
            "Insufficient terminal node-graph byte workspace."
        );
    }
    using Workspace = TerminalNodeGraphWorkspace<NodePolicy>;
    auto* base = static_cast<unsigned char*>(storage);
    return {
        reinterpret_cast<typename Workspace::NodeValue*>(
            base + layout.node_values_offset
        ),
        layout.capacities.node_values,
        reinterpret_cast<typename Workspace::NodeMetadata*>(
            base + layout.node_metadata_offset
        ),
        layout.capacities.node_metadata,
        reinterpret_cast<typename Workspace::NodeIndices*>(
            base + layout.node_indices_offset
        ),
        layout.capacities.node_indices,
        reinterpret_cast<std::uint8_t*>(base + layout.row_status_offset),
        layout.capacities.row_status,
        reinterpret_cast<reductions::MomentSums*>(
            base + layout.thread_moments_offset
        ),
        layout.capacities.thread_moments,
    };
}

template<typename NodePolicy>
void validate_terminal_node_graph_workspace(
    TerminalNodeGraphWorkspace<NodePolicy> workspace,
    TerminalNodeGraphWorkspaceRequirements required
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
    validate(
        workspace.node_values,
        workspace.node_value_capacity,
        required.node_values,
        "terminal node values"
    );
    validate(
        workspace.node_metadata,
        workspace.node_metadata_capacity,
        required.node_metadata,
        "terminal node metadata"
    );
    validate(
        workspace.node_indices,
        workspace.node_index_capacity,
        required.node_indices,
        "terminal node indices"
    );
    validate(
        workspace.row_status,
        workspace.row_status_capacity,
        required.row_status,
        "terminal node row status"
    );
    validate(
        workspace.thread_moments,
        workspace.thread_moment_capacity,
        required.thread_moments,
        "terminal node thread moments"
    );
}

static_assert(std::is_trivially_copyable_v<SensitivityNodeIndices<4U>>);
static_assert(std::is_trivially_copyable_v<TerminalNodeGraphConfiguration>);

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
