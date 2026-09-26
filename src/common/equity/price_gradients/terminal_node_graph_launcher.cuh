// Shared host composition and bounded workspace planning for terminal node graphs.
#pragma once

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_terminal_node_graph.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

struct TerminalNodeGraphExecutionPlan {
    mcpg::TerminalNodeGraphConfiguration graph{};
    mcpg::TerminalNodeGraphWorkspaceLayout workspace{};
};

namespace terminal_node_graph_detail {

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
    check_cuda(cudaGetDevice(&device), "terminal node-graph CUDA device");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "terminal node-graph CUDA device properties"
    );
    if (properties.multiProcessorCount <= 0) {
        throw std::runtime_error(
            "Terminal node graph requires a CUDA multiprocessor."
        );
    }
    return static_cast<unsigned int>(properties.multiProcessorCount);
}

}  // namespace terminal_node_graph_detail

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
TerminalNodeGraphExecutionPlan make_terminal_node_graph_execution_plan(
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
            "Invalid terminal node-graph execution dimensions."
        );
    }

    using NodePolicy = mcpg::SelectedTerminalNodePolicy<
        Dynamics, ProductPolicy, typename HostPlan::Preparation
    >;
    auto rows = std::min(
        launch.result_count,
        Tuning::kMaximumRowChunkSize
    );
    auto paths = terminal_node_graph_detail::whole_reduction_thread_chunk(
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
                terminal_node_graph_detail::whole_reduction_thread_chunk(
                    graph.path_chunk_size / 2U,
                    launch.threads_per_block
                );
        } else {
            throw std::invalid_argument(
                "Terminal node graph exceeds its workspace budget."
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
        terminal_node_graph_detail::current_multiprocessor_count();
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
            "Terminal node-graph path-shard count overflow."
        );
    }
    graph.path_shards = static_cast<unsigned int>(shards);
    return {graph, layout};
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
std::size_t terminal_node_graph_workspace_bytes(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    if (host.sensitivity_count() == 0U) return 0U;
    return make_terminal_node_graph_execution_plan<
        Orders,
        Dynamics,
        ProductPolicy,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(host, launch).workspace.bytes;
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan,
    typename PriceOnlyLaunch>
void launch_terminal_node_graph_sensitivities(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace_storage,
    std::size_t workspace_bytes,
    PriceOnlyLaunch&& launch_price_only,
    const char* kernel_name,
    const char* variant
) {
    static_assert(pg::requests_second_v<Orders>);
    validate_terminal_diagonal_sensitivity_launch<Orders>(
        host, device, stencil_outputs, configuration, outputs
    );
    if (host.sensitivity_count() == 0U) {
        std::forward<PriceOnlyLaunch>(launch_price_only)();
        return;
    }

    using NodePolicy = mcpg::SelectedTerminalNodePolicy<
        Dynamics, ProductPolicy, typename HostPlan::Preparation
    >;
    const auto execution = make_terminal_node_graph_execution_plan<
        Orders,
        Dynamics,
        ProductPolicy,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(host, configuration);
    const auto workspace = mcpg::make_terminal_node_graph_workspace<
        NodePolicy
    >(workspace_storage, workspace_bytes, execution.workspace);
    mcpg::launch_device_prepared_terminal_node_graph<
        Orders,
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(
        device,
        mcpg::make_device_prepared_plan(host),
        configuration,
        execution.graph,
        workspace,
        outputs,
        stencil_outputs,
        kernel_name,
        variant
    );
}

}  // namespace ai_factory::workbench::equity::price_gradients
