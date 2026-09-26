// Shared host composition and workspace planning for path node graphs.
#pragma once

#include "common/equity/price_gradients/node_graph_execution_plan.hpp"
#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_path_node_graph.cuh"

#include <cstddef>
#include <utility>

namespace ai_factory::workbench::equity::price_gradients {

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Schedule,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
std::size_t path_node_graph_workspace_bytes(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    if (host.sensitivity_count() == 0U) return 0U;
    using NodePolicy = mcpg::PathNodePolicy<
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule
    >;
    return make_node_graph_execution_plan<
        Orders,
        NodePolicy,
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
    typename Schedule,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan,
    typename PriceOnlyLaunch>
void launch_path_node_graph_sensitivities(
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

    using NodePolicy = mcpg::PathNodePolicy<
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule
    >;
    const auto execution = make_node_graph_execution_plan<
        Orders,
        NodePolicy,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(host, configuration);
    const auto workspace = mcpg::make_terminal_node_graph_workspace<
        NodePolicy
    >(workspace_storage, workspace_bytes, execution.workspace);
    mcpg::launch_device_prepared_path_node_graph<
        Orders,
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule,
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
