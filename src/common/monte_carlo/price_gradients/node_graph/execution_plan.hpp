// GPU-aware bounded workspace planning shared by Monte Carlo node graphs.
#pragma once

#include "common/check_cuda.cuh"
#include "common/monte_carlo/price_gradients/node_graph/capacity.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"
#include "common/price_gradients/launch.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct NodeGraphExecutionPlan {
    TerminalNodeGraphConfiguration graph{};
    TerminalNodeGraphWorkspaceLayout workspace{};
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

inline std::size_t whole_reduction_thread_capacity(
    std::size_t requested,
    unsigned int reduction_threads
) {
    const auto threads = static_cast<std::size_t>(reduction_threads);
    const auto remainder = requested % threads;
    if (remainder == 0U) return requested;
    const auto padding = threads - remainder;
    if (requested > std::numeric_limits<std::size_t>::max() - padding) {
        throw std::overflow_error(
            "Sensitivity node-graph path capacity overflow."
        );
    }
    return requested + padding;
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
    const pg::LaunchConfiguration& launch,
    bool preserve_complete_path_reduction = false
) {
    static_assert(pg::requests_second_v<Orders>);
    static_assert(tuning::valid_node_profile_v<Tuning>);
    constexpr auto node_capacity =
        terminal_node_graph_node_capacity<MaximumSensitivities>();
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
    auto paths = preserve_complete_path_reduction
        ? node_graph_execution_detail::whole_reduction_thread_capacity(
            launch.paths_per_price,
            launch.threads_per_block
        )
        : node_graph_execution_detail::whole_reduction_thread_chunk(
            std::min(Tuning::kPathChunkSize, launch.paths_per_price),
            launch.threads_per_block
        );
    TerminalNodeGraphConfiguration graph{rows, paths, 1U};
    auto layout = terminal_node_graph_workspace_layout<
        NodePolicy, Orders, MaximumSensitivities
    >(host.sensitivity_count(), launch.threads_per_block, graph);
    while (layout.bytes > Tuning::kWorkspaceByteLimit) {
        if (graph.row_chunk_size > 1U) {
            graph.row_chunk_size = (graph.row_chunk_size + 1U) / 2U;
        } else if (
            !preserve_complete_path_reduction
            && graph.path_chunk_size
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
        layout = terminal_node_graph_workspace_layout<
            NodePolicy, Orders, MaximumSensitivities
        >(host.sensitivity_count(), launch.threads_per_block, graph);
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


template<typename WorkspaceLayout>
struct BasicMixedNodeGraphExecutionPlan {
    TerminalNodeGraphConfiguration graph{};
    WorkspaceLayout workspace{};
};

using MixedNodeGraphExecutionPlan = BasicMixedNodeGraphExecutionPlan<
    MixedNodeGraphWorkspaceLayout
>;

template<
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename HostPlan,
    typename MakeWorkspaceLayout>
auto make_mixed_node_graph_execution_plan_with_layout(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch,
    MakeWorkspaceLayout make_workspace_layout
) {
    static_assert(tuning::valid_node_profile_v<Tuning>);
    constexpr auto node_capacity =
        node_graph_detail::mixed_node_graph_node_capacity<
            MaximumSensitivities, MaximumMixedSensitivities
        >();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);

    const auto& sensitivity_graph = host.sensitivity_graph;
    if (host.sensitivity_count() == 0U
        || host.sensitivity_count() > MaximumSensitivities
        || sensitivity_graph.mixed_second.empty()
        || sensitivity_graph.mixed_second.size()
            > MaximumMixedSensitivities
        || sensitivity_graph.node_capacity > node_capacity
        || launch.result_count == 0U
        || launch.paths_per_price < 2U) {
        throw std::invalid_argument(
            "Invalid mixed sensitivity node-graph execution dimensions."
        );
    }

    auto rows = std::min(
        launch.result_count,
        Tuning::kMaximumRowChunkSize
    );
    auto paths = node_graph_execution_detail::whole_reduction_thread_chunk(
        std::min(Tuning::kPathChunkSize, launch.paths_per_price),
        launch.threads_per_block
    );
    TerminalNodeGraphConfiguration graph{rows, paths, 1U};
    auto layout = make_workspace_layout(graph);
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
                "Mixed sensitivity node graph exceeds its workspace budget."
            );
        }
        layout = make_workspace_layout(graph);
    }

    constexpr auto teams_per_block =
        Tuning::kThreadsPerBlock / GroupSize;
    const auto multiprocessors =
        node_graph_execution_detail::current_multiprocessor_count();
    const auto target_blocks = static_cast<std::size_t>(multiprocessors)
        * Tuning::kResidentWaves;
    const auto desired_shards =
        (target_blocks + graph.row_chunk_size - 1U)
        / graph.row_chunk_size;
    const auto useful_shards =
        (graph.path_chunk_size + teams_per_block - 1U)
        / teams_per_block;
    const auto shards = std::max<std::size_t>(
        1U, std::min(desired_shards, useful_shards)
    );
    if (shards > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error(
            "Mixed sensitivity node-graph path-shard count overflow."
        );
    }
    graph.path_shards = static_cast<unsigned int>(shards);
    return BasicMixedNodeGraphExecutionPlan<decltype(layout)>{
        graph, layout
    };
}

template<
    typename NodePolicy,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename HostPlan>
MixedNodeGraphExecutionPlan make_mixed_node_graph_execution_plan(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    const auto make_layout = [&](TerminalNodeGraphConfiguration graph) {
        return mixed_node_graph_workspace_layout<NodePolicy>(
            host.sensitivity_count(),
            host.sensitivity_graph,
            launch.threads_per_block,
            graph
        );
    };
    return make_mixed_node_graph_execution_plan_with_layout<
        MaximumSensitivities,
        MaximumMixedSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(host, launch, make_layout);
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
