// Compose compact sensitivity inputs with the common frozen-exercise LSM engine.
#pragma once

#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <algorithm>
#include <concepts>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename ScheduleTime>
ScheduleTime make_schedule_time(pg::TimeConfiguration time) {
    if constexpr (std::same_as<
                      ScheduleTime,
                      simulation::FixedStepTimeConfiguration>) {
        return {time.dt, time.simulation_steps_per_day};
    } else {
        static_assert(std::same_as<
            ScheduleTime,
            simulation::ExactTransitionTimeConfiguration
        >);
        return {
            time.dt * static_cast<float>(time.simulation_steps_per_day)
        };
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Regressor,
    typename HostPlan
>
longstaff_schwartz::LaunchResult launch_device_prepared_sensitivities(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* product_name
) {
    if (host.request.orders != Orders) {
        throw std::invalid_argument(
            "Sensitivity request and frozen-exercise kernel order differ."
        );
    }
    if (launch.method != pg::PricingMethod::monte_carlo) {
        throw std::invalid_argument(
            "Frozen-exercise sensitivities require Monte Carlo."
        );
    }
    if (launch.sensitivity_batch_size != 1U) {
        throw std::invalid_argument(
            "Frozen-exercise sensitivities require B=1."
        );
    }
    const std::size_t rows = host.result_count;
    const std::size_t selected = host.sensitivity_count();
    if (outputs.price_capacity < rows) {
        throw std::invalid_argument(
            "Insufficient frozen-exercise price output capacity."
        );
    }
    std::vector<pg::BufferRange> output_ranges{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
        pg::checked_buffer_range(
            outputs.price_standard_errors, rows, sizeof(float)
        ),
    };
    if (selected != 0U) {
        if (outputs.sensitivity_capacity < rows * selected) {
            throw std::invalid_argument(
                "Insufficient frozen-exercise sensitivity output capacity."
            );
        }
        if constexpr (pg::requests_first_v<Orders>) {
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.gradients, rows * selected, sizeof(float)
            ));
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.gradient_standard_errors,
                rows * selected,
                sizeof(float)
            ));
        }
        if constexpr (pg::requests_second_v<Orders>) {
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.diagonal_hessians,
                rows * selected,
                sizeof(float)
            ));
            output_ranges.push_back(pg::checked_buffer_range(
                outputs.diagonal_hessian_standard_errors,
                rows * selected,
                sizeof(float)
            ));
        }
    }
    // The shared compact-buffer validator uses block_count as a total task
    // cap. LSM uses it as blocks per price, so validate its buffers with the
    // equivalent legal compact geometry and retain the caller's LSM geometry
    // for the actual launch below.
    pg::LaunchConfiguration compact_validation = launch;
    compact_validation.block_count = launch.result_count
        * std::max<std::size_t>(selected, 1U);
    mcpg::validate_device_prepared_launch(
        device,
        mcpg::make_device_prepared_plan(host),
        stencil_outputs,
        compact_validation,
        output_ranges
    );

    const typename Policy::HostInputs host_inputs =
        Policy::make_host_inputs(host, launch.result_offset);
    const typename Policy::DeviceInputs device_inputs =
        Policy::make_device_inputs(
            host,
            device,
            stencil_outputs,
            launch.result_offset,
            outputs
        );
    return longstaff_schwartz::launch_longstaff_schwartz_cuda<
        Policy,
        Regressor
    >(
        device_inputs,
        host_inputs,
        launch.result_count,
        launch.paths_per_price,
        make_schedule_time<typename Policy::TimeConfiguration>(host.time),
        launch.threads_per_block,
        launch.block_count,
        launch.base_seed + launch.result_offset,
        outputs.prices,
        outputs.price_standard_errors,
        diagnostic_name,
        diagnostic_variant,
        product_name
    );
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
