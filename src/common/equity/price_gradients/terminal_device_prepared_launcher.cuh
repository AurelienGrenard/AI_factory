// Shared host composition for device-prepared terminal Monte Carlo sensitivities.
#pragma once

#include "common/monte_carlo/price_gradients/device_prepared_terminal_diagonal_kernel.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_terminal_kernel.cuh"
#include "common/equity/price_gradients/device_prepared_stencil_launcher.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"

#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename HostPlan>
void validate_terminal_first_sensitivity_launch(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs
) {
    const auto rows = host.result_count;
    const auto sensitivity_count = host.sensitivity_count();
    if (host.request.orders != pg::SensitivityOrders::first) {
        throw std::invalid_argument(
            "The price-gradient launcher requires a first-order request."
        );
    }
    if (outputs.price_capacity < rows
        || outputs.gradient_capacity < rows * sensitivity_count) {
        throw std::invalid_argument(
            "Insufficient terminal sensitivity output capacity."
        );
    }
    std::vector<pg::BufferRange> output_ranges{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
        pg::checked_buffer_range(
            outputs.price_standard_errors, rows, sizeof(float)
        ),
    };
    if (sensitivity_count != 0U) {
        output_ranges.push_back(pg::checked_buffer_range(
            outputs.gradients, rows * sensitivity_count, sizeof(float)
        ));
        output_ranges.push_back(pg::checked_buffer_range(
            outputs.gradient_standard_errors,
            rows * sensitivity_count,
            sizeof(float)
        ));
    }
    mcpg::validate_device_prepared_launch(
        device,
        mcpg::make_device_prepared_plan(host),
        stencil_outputs,
        launch,
        output_ranges
    );
}

template<pg::SensitivityOrders Orders, typename HostPlan>
void validate_terminal_diagonal_sensitivity_launch(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    const auto rows = host.result_count;
    const auto selected = host.sensitivity_count();
    const auto sensitivity_count = rows * selected;
    if (host.request.orders != Orders) {
        throw std::invalid_argument(
            "Sensitivity request and terminal kernel order differ."
        );
    }
    if (outputs.prices == nullptr
        || outputs.price_standard_errors == nullptr
        || outputs.price_capacity < rows) {
        throw std::invalid_argument("Insufficient terminal price outputs.");
    }
    if constexpr (pg::requests_first_v<Orders>) {
        if (outputs.gradients == nullptr
            || outputs.gradient_standard_errors == nullptr
            || outputs.sensitivity_capacity < sensitivity_count) {
            throw std::invalid_argument(
                "Insufficient terminal gradient outputs."
            );
        }
    }
    if (selected != 0U
        && (outputs.diagonal_hessians == nullptr
            || outputs.diagonal_hessian_standard_errors == nullptr
            || outputs.sensitivity_capacity < sensitivity_count)) {
        throw std::invalid_argument(
            "Insufficient terminal diagonal-Hessian outputs."
        );
    }
    std::vector<pg::BufferRange> output_ranges{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
        pg::checked_buffer_range(
            outputs.price_standard_errors, rows, sizeof(float)
        ),
    };
    if constexpr (pg::requests_first_v<Orders>) {
        if (selected != 0U) {
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.gradients, sensitivity_count, sizeof(float)
            ));
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.gradient_standard_errors,
                sensitivity_count,
                sizeof(float)
            ));
        }
    }
    if (selected != 0U) {
        output_ranges.push_back(pg::checked_buffer_range(
            outputs.diagonal_hessians, sensitivity_count, sizeof(float)
        ));
        output_ranges.push_back(pg::checked_buffer_range(
            outputs.diagonal_hessian_standard_errors,
            sensitivity_count,
            sizeof(float)
        ));
    }
    mcpg::validate_device_prepared_launch(
        device,
        mcpg::make_device_prepared_plan(host),
        stencil_outputs,
        launch,
        output_ranges
    );
}

template<pg::SensitivityOrders Orders, typename HostPlan>
void prepare_terminal_sensitivity_stencils(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count,
    const char* kernel_name
) {
    prepare_device_sensitivity_stencils<Orders>(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        kernel_name
    );
}

template<typename HostPlan>
void prepare_terminal_first_sensitivity_stencils(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count,
    const char* kernel_name
) {
    prepare_terminal_sensitivity_stencils<pg::SensitivityOrders::first>(
        host, device, stencil_outputs, result_offset, result_count,
        kernel_name
    );
}

template<
    typename Dynamics,
    typename ProductPolicy,
    typename HostPlan,
    typename PriceOnlyLaunch
>
void launch_terminal_first_sensitivities(
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
    mcpg::launch_device_prepared_terminal_first_sensitivities<
        Dynamics,
        ProductPolicy,
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
    typename HostPlan,
    typename PriceOnlyLaunch
>
void launch_terminal_diagonal_sensitivities(
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
    mcpg::launch_device_prepared_terminal_diagonal_sensitivities<
        Orders,
        Dynamics,
        ProductPolicy,
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

}  // namespace ai_factory::workbench::equity::price_gradients
