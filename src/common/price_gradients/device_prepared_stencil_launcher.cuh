// Shared host launcher for represented stencils built from compact inputs.
#pragma once

#include "common/monte_carlo/price_gradients/device_prepared_stencil_kernel.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"

#include <span>
#include <stdexcept>

namespace ai_factory::workbench::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<pg::SensitivityOrders Orders, typename HostPlan>
void prepare_device_sensitivity_stencils(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count,
    const char* kernel_name
) {
    if (host.request.orders != Orders) {
        throw std::invalid_argument(
            "Sensitivity request and stencil preparation order differ."
        );
    }
    const auto plan = mcpg::make_device_prepared_plan(host);
    if (result_count == 0U || host.sensitivity_count() == 0U) return;
    const pg::LaunchConfiguration validation_launch{
        pg::PricingMethod::monte_carlo,
        result_offset,
        result_count,
        2U,
        256U,
        result_count * host.sensitivity_count(),
        0U,
        1U,
    };
    mcpg::validate_device_prepared_launch(
        device,
        plan,
        stencil_outputs,
        validation_launch,
        std::span<const pg::BufferRange>{}
    );
    mcpg::launch_device_prepared_stencils<
        Orders,
        typename HostPlan::DeviceInputs
    >(
        device,
        plan,
        result_offset,
        result_count,
        stencil_outputs,
        kernel_name
    );
}

}  // namespace ai_factory::workbench::price_gradients
