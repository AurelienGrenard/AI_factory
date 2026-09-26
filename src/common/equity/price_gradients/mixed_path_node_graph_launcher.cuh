// Public host composition for path sensitivity graphs with mixed outputs.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/monte_carlo/price_gradients/node_graph/execution_plan.hpp"
#include "common/monte_carlo/price_gradients/path_node_graph/mixed_launcher.cuh"

#include <cstddef>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Schedule,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int TeamSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
auto make_mixed_path_node_graph_execution_plan(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    using NodePolicy = mcpg::PathNodePolicy<
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule
    >;
    const auto make_layout = [&](mcpg::TerminalNodeGraphConfiguration graph) {
        return mcpg::mixed_path_node_graph_workspace_layout<
            NodePolicy, Dynamics, Schedule
        >(
            host.sensitivity_count(),
            host.sensitivity_graph,
            launch.threads_per_block,
            graph
        );
    };
    return mcpg::make_mixed_node_graph_execution_plan_with_layout<
        MaximumSensitivities,
        MaximumMixedSensitivities,
        TeamSize,
        NodesPerWorker,
        Tuning
    >(host, launch, make_layout);
}

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Schedule,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int TeamSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
std::size_t mixed_path_node_graph_workspace_bytes(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    return make_mixed_path_node_graph_execution_plan<
        Dynamics,
        ProductPolicy,
        Schedule,
        MaximumSensitivities,
        MaximumMixedSensitivities,
        TeamSize,
        NodesPerWorker,
        Tuning
    >(host, launch).workspace.bytes;
}

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Schedule,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int TeamSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
void launch_mixed_path_node_graph_sensitivities(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    typename HostPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace_storage,
    std::size_t workspace_bytes,
    const char* kernel_name,
    const char* variant
) {
    using NodePolicy = mcpg::PathNodePolicy<
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule
    >;
    const auto execution = make_mixed_path_node_graph_execution_plan<
        Dynamics,
        ProductPolicy,
        Schedule,
        MaximumSensitivities,
        MaximumMixedSensitivities,
        TeamSize,
        NodesPerWorker,
        Tuning
    >(host, launch);
    const auto workspace = mcpg::make_mixed_path_node_graph_workspace<
        NodePolicy, Dynamics, Schedule
    >(workspace_storage, workspace_bytes, execution.workspace);
    const auto device_graph = mcpg::upload_mixed_sensitivity_graph(
        workspace.graph, host.sensitivity_graph
    );
    mcpg::launch_device_prepared_path_mixed_node_graph<
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule,
        MaximumSensitivities,
        MaximumMixedSensitivities,
        TeamSize,
        NodesPerWorker,
        Tuning
    >(
        device,
        mcpg::make_device_prepared_plan(host),
        host.sensitivity_graph,
        device_graph,
        launch,
        execution.graph,
        workspace,
        outputs,
        mixed_outputs,
        stencil_outputs,
        mixed_stencil_outputs,
        kernel_name,
        variant
    );
}

}  // namespace ai_factory::workbench::equity::price_gradients
