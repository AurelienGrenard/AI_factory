// Compose compact sensitivity inputs with the common frozen-exercise LSM engine.
#pragma once

#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_exercise_node_graph/value_policy.cuh"
#include "common/monte_carlo/price_gradients/node_graph/execution_plan.hpp"
#include "common/monte_carlo/price_gradients/node_graph/mixed_launch_validation.hpp"
#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <algorithm>
#include <concepts>
#include <span>
#include <stdexcept>
#include <utility>
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

namespace launcher_detail {

template<pg::SensitivityOrders Orders, typename HostPlan>
void validate_launch(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    mcpg::DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
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
}

template<
    typename Policy,
    typename Regressor,
    typename HostPlan,
    typename DeviceInputsBuilder>
longstaff_schwartz::LaunchResult launch_with_device_inputs(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    DeviceInputsBuilder&& build_device_inputs,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* product_name
) {
    const typename Policy::HostInputs host_inputs =
        Policy::make_host_inputs(host, launch.result_offset);
    const typename Policy::DeviceInputs device_inputs =
        std::forward<DeviceInputsBuilder>(build_device_inputs)();
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

}  // namespace launcher_detail

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Regressor,
    typename HostPlan>
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
    launcher_detail::validate_launch<Orders>(
        host, device, stencil_outputs, launch, outputs
    );
    return launcher_detail::launch_with_device_inputs<Policy, Regressor>(
        host,
        launch,
        outputs,
        [&] {
            return Policy::make_device_inputs(
                host,
                device,
                stencil_outputs,
                launch.result_offset,
                outputs
            );
        },
        diagnostic_name,
        diagnostic_variant,
        product_name
    );
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
std::size_t frozen_exercise_node_graph_workspace_bytes(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    if (host.sensitivity_count() == 0U) return 0U;
    return mcpg::make_node_graph_execution_plan<
        Orders,
        FrozenExerciseValueNodePolicy,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(host, launch, true).workspace.bytes;
}

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    typename Regressor,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
longstaff_schwartz::LaunchResult
launch_device_prepared_node_graph_sensitivities(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    void* workspace_storage,
    std::size_t workspace_bytes,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* product_name
) {
    static_assert(pg::requests_second_v<Orders>);
    launcher_detail::validate_launch<Orders>(
        host, device, stencil_outputs, launch, outputs
    );

    mcpg::TerminalNodeGraphConfiguration graph{
        1U,
        std::max<std::size_t>(launch.threads_per_block, 1U),
        1U,
    };
    mcpg::TerminalNodeGraphWorkspace<
        FrozenExerciseValueNodePolicy
    > workspace{};
    if (host.sensitivity_count() != 0U) {
        const auto execution = mcpg::make_node_graph_execution_plan<
            Orders,
            FrozenExerciseValueNodePolicy,
            MaximumSensitivities,
            GroupSize,
            NodesPerWorker,
            Tuning
        >(host, launch, true);
        graph = execution.graph;
        workspace = mcpg::make_terminal_node_graph_workspace<
            FrozenExerciseValueNodePolicy
        >(workspace_storage, workspace_bytes, execution.workspace);
    }

    return launcher_detail::launch_with_device_inputs<Policy, Regressor>(
        host,
        launch,
        outputs,
        [&] {
            return Policy::make_device_inputs(
                host,
                device,
                stencil_outputs,
                launch.result_offset,
                outputs,
                graph,
                workspace
            );
        },
        diagnostic_name,
        diagnostic_variant,
        product_name
    );
}


template<
    typename Policy,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
std::size_t frozen_exercise_mixed_node_graph_workspace_bytes(
    const HostPlan& host,
    const pg::LaunchConfiguration& launch
) {
    const auto make_layout = [&](auto configuration) {
        return Policy::make_mixed_workspace_layout(
            host, launch.threads_per_block, configuration
        );
    };
    return mcpg::make_mixed_node_graph_execution_plan_with_layout<
        MaximumSensitivities,
        MaximumMixedSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(host, launch, make_layout).workspace.bytes;
}

template<
    typename Policy,
    typename Regressor,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = mcpg::tuning::DefaultTerminalNodeTuning,
    typename HostPlan>
longstaff_schwartz::LaunchResult
launch_device_prepared_mixed_node_graph_sensitivities(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    typename HostPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace_storage,
    std::size_t workspace_bytes,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* product_name
) {
    if (launch.method != pg::PricingMethod::monte_carlo) {
        throw std::invalid_argument(
            "Frozen-exercise mixed sensitivities require Monte Carlo."
        );
    }
    if (launch.sensitivity_batch_size != 1U) {
        throw std::invalid_argument(
            "Frozen-exercise mixed sensitivities require B=1."
        );
    }

    const auto make_layout = [&](auto configuration) {
        return Policy::make_mixed_workspace_layout(
            host, launch.threads_per_block, configuration
        );
    };
    const auto execution =
        mcpg::make_mixed_node_graph_execution_plan_with_layout<
            MaximumSensitivities,
            MaximumMixedSensitivities,
            GroupSize,
            NodesPerWorker,
            Tuning
        >(host, launch, make_layout);
    const auto workspace = Policy::make_mixed_workspace(
        workspace_storage, workspace_bytes, execution.workspace
    );
    const auto device_graph = mcpg::upload_mixed_sensitivity_graph(
        workspace.graph, host.sensitivity_graph
    );
    mcpg::validate_mixed_node_graph_launch(
        device,
        mcpg::make_device_prepared_plan(host),
        host.sensitivity_graph,
        device_graph,
        launch,
        outputs,
        mixed_outputs,
        stencil_outputs,
        mixed_stencil_outputs
    );
    Policy::validate_mixed_workspace(
        workspace, execution.workspace.capacities
    );

    return launcher_detail::launch_with_device_inputs<Policy, Regressor>(
        host,
        launch,
        outputs,
        [&] {
            return Policy::make_device_inputs(
                host,
                device,
                stencil_outputs,
                mixed_stencil_outputs,
                launch.result_offset,
                outputs,
                mixed_outputs,
                execution.graph,
                device_graph,
                workspace
            );
        },
        diagnostic_name,
        diagnostic_variant,
        product_name
    );
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
