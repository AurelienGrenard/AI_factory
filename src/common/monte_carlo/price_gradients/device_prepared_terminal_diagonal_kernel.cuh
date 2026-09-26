// Terminal composition over the generic diagonal Monte Carlo kernel.
#pragma once

#include "common/monte_carlo/price_gradients/device_prepared_diagonal_kernel.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"

namespace ai_factory::workbench::monte_carlo::price_gradients {

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Tuning = tuning::DefaultTerminalMonoTuning,
    typename Inputs>
void launch_device_prepared_terminal_diagonal_sensitivities(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
    using Policy = TerminalSensitivityPolicy<
        Orders, Dynamics, ProductPolicy, Preparation
    >;
    launch_device_prepared_diagonal_sensitivities<
        Orders, Policy, Preparation, Tuning
    >(
        inputs,
        plan,
        configuration,
        outputs,
        stencil_outputs,
        kernel_name,
        variant
    );
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
