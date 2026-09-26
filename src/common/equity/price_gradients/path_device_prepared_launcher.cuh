// Host composition for device-prepared path Monte Carlo sensitivities.
#pragma once

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_diagonal_kernel.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_first_sensitivity_kernel.cuh"
#include "common/monte_carlo/price_gradients/path_sensitivity_policy.cuh"

#include <utility>

namespace ai_factory::workbench::equity::price_gradients {

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Schedule,
    typename HostPlan,
    typename PriceOnlyLaunch>
void launch_path_first_sensitivities(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs,
    PriceOnlyLaunch&& launch_price_only,
    const char* kernel_name,
    const char* variant
) {
    validate_terminal_first_sensitivity_launch(
        host, device, stencil_outputs, configuration, outputs
    );
    if (host.sensitivity_count() == 0U) {
        std::forward<PriceOnlyLaunch>(launch_price_only)();
        return;
    }
    using Policy = mcpg::FirstPathSensitivityPolicy<
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule
    >;
    mcpg::launch_device_prepared_first_sensitivities<
        Policy,
        typename HostPlan::Preparation
    >(
        device,
        mcpg::make_device_prepared_plan(host),
        configuration,
        outputs,
        stencil_outputs,
        kernel_name,
        variant
    );
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Schedule,
    typename MonoTuning = mcpg::tuning::DefaultTerminalMonoTuning,
    typename HostPlan,
    typename PriceOnlyLaunch>
void launch_path_diagonal_sensitivities(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
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
    using Policy = mcpg::PathSensitivityPolicy<
        Orders,
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        Schedule
    >;
    mcpg::launch_device_prepared_diagonal_sensitivities<
        Orders,
        Policy,
        typename HostPlan::Preparation,
        MonoTuning
    >(
        device,
        mcpg::make_device_prepared_plan(host),
        configuration,
        outputs,
        stencil_outputs,
        kernel_name,
        variant
    );
}

}  // namespace ai_factory::workbench::equity::price_gradients
