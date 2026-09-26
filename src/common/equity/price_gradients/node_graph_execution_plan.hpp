// GPU-aware bounded-workspace planner shared by sensitivity node graphs.
#pragma once

#include "common/check_cuda.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"
#include "common/price_gradients/launch.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

struct NodeGraphExecutionPlan {
    mcpg::TerminalNodeGraphConfiguration graph{};
    mcpg::TerminalNodeGraphWorkspaceLayout workspace{};
};

namespace node_graph_execution_detail {

inline std::size_t whole_reduction_thread_chunk(
    std::size_t requested,
    unsigned int reduction_threads
) {
    const auto threads = static_cast<std::size_t>(reduction_threads);
    if (requested < threads) return threads;
    return requested - requested % threads;
}

inline unsigned int current_multiprocessor_count() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "sensitivity node-graph CUDA device");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "sensitivity node-graph CUDA device properties"
    );
    if (properties.multiProcessorCount <= 0) {
        throw std::runtime_error(
            "Sensitivity node graph requires a CUDA multiprocessor."
        );
    }
    return static_cast<unsigned int>(properties.multiProcessorCount);
}

}  // namespace node_graph_execution_detail

template<
    pg::SensitivityOrders Orders,
    typename NodePolicy,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename HostPlan>
NodeGraphExecutionPlan make_node_graph_execution_plan(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    static_assert(pg::requests_second_v<Orders>);
    static_assert(mcpg::tuning::valid_node_profile_v<Tuning>);
    constexpr auto node_capacity =
        mcpg::terminal_node_graph_node_capacity<MaximumSensitivities>();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);

    if (host.sensitivity_count() == 0U
        || host.sensitivity_count() > MaximumSensitivities
        || launch.result_count == 0U
        || launch.paths_per_price < 2U) {
        throw std::invalid_argument(
            "Invalid sensitivity node-graph execution dimensions."
        );
    }

    auto rows = std::min(
        launch.result_count,
        Tuning::kMaximumRowChunkSize
    );
    auto paths =
        node_graph_execution_detail::whole_reduction_thread_chunk(
            std::min(Tuning::kPathChunkSize, launch.paths_per_price),
            launch.threads_per_block
        );
    mcpg::TerminalNodeGraphConfiguration graph{rows, paths, 1U};
    auto layout = mcpg::terminal_node_graph_workspace_layout<
        NodePolicy, Orders, MaximumSensitivities
    >(
        host.sensitivity_count(), launch.threads_per_block, graph
    );
    while (layout.bytes > Tuning::kWorkspaceByteLimit) {
        if (graph.row_chunk_size > 1U) {
            graph.row_chunk_size = (graph.row_chunk_size + 1U) / 2U;
        } else if (
            graph.path_chunk_size
                > static_cast<std::size_t>(launch.threads_per_block)
        ) {
            graph.path_chunk_size =
                node_graph_execution_detail::whole_reduction_thread_chunk(
                    graph.path_chunk_size / 2U,
                    launch.threads_per_block
                );
        } else {
            throw std::invalid_argument(
                "Sensitivity node graph exceeds its workspace budget."
            );
        }
        layout = mcpg::terminal_node_graph_workspace_layout<
            NodePolicy, Orders, MaximumSensitivities
        >(
            host.sensitivity_count(), launch.threads_per_block, graph
        );
    }

    constexpr auto groups_per_block =
        Tuning::kThreadsPerBlock / GroupSize;
    const auto multiprocessors =
        node_graph_execution_detail::current_multiprocessor_count();
    const auto target_blocks = static_cast<std::size_t>(multiprocessors)
        * Tuning::kResidentWaves;
    const auto desired_shards =
        (target_blocks + graph.row_chunk_size - 1U)
        / graph.row_chunk_size;
    const auto useful_shards =
        (graph.path_chunk_size + groups_per_block - 1U)
        / groups_per_block;
    const auto shards = std::max<std::size_t>(
        1U, std::min(desired_shards, useful_shards)
    );
    if (shards > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error(
            "Sensitivity node-graph path-shard count overflow."
        );
    }
    graph.path_shards = static_cast<unsigned int>(shards);
    return {graph, layout};
}

}  // namespace ai_factory::workbench::equity::price_gradients
