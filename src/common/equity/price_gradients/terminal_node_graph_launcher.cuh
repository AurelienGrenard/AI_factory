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

using TerminalNodeGraphExecutionPlan = mcpg::NodeGraphExecutionPlan;

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
    using NodePolicy = mcpg::SelectedTerminalNodePolicy<
        Dynamics, ProductPolicy, typename HostPlan::Preparation>;
    return mcpg::make_node_graph_execution_plan<
        Orders, NodePolicy, MaximumSensitivities, GroupSize,
        NodesPerWorker, Tuning>(host, launch);
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
    static_assert(Orders != pg::SensitivityOrders::none);
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
