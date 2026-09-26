// One thread prepares and evaluates one selected closed-form Hessian graph.
#pragma once

#include "common/closed_form/price_gradients/mixed_workspace.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_launch_validation.hpp"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/mixed_sensitivity_preparation.cuh"
#include "common/price_gradients/mixed_sensitivity_reconstruction.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::closed_form::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

namespace mixed_detail {

template<std::size_t MaximumSensitivities>
struct AxisValues {
    pg::SensitivityStencil<4U> stencils[MaximumSensitivities]{};
    pg::SensitivityValues<4U> values[MaximumSensitivities]{};
};

template<
    typename Preparation,
    std::size_t MaximumSensitivities,
    typename Inputs,
    typename Policy>
__device__ __forceinline__ bool prepare_and_evaluate_axes(
    Inputs inputs,
    mcpg::DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph graph,
    const typename Preparation::Scenario& central,
    float central_value,
    std::size_t row,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    AxisValues<MaximumSensitivities>& axes
) {
    for (std::size_t sensitivity = 0U;
         sensitivity < plan.sensitivity_count;
         ++sensitivity) {
        const auto use = graph.coordinate_uses[sensitivity];
        if (use == pg::SensitivityCoordinateUse::none) continue;

        typename Preparation::Scenario nodes[4U]{};
        int error = preparation::valid;
        if (!pg::prepare_axis_sensitivity_nodes<Preparation>(
                central,
                inputs.sensitivities[sensitivity],
                use,
                plan.time,
                axes.stencils[sensitivity],
                nodes,
                error
            )) {
            preparation::record_error(
                stencil_outputs.error, error, row, sensitivity
            );
            return false;
        }

        auto& values = axes.values[sensitivity];
        values[0U] = central_value;
        for (std::size_t node = 1U;
             node < pg::active_node_count(axes.stencils[sensitivity]);
             ++node) {
            values[node] = Policy::evaluate(nodes[node]);
        }
        stencil_outputs.stencils[
            row * plan.sensitivity_count + sensitivity
        ] = axes.stencils[sensitivity];
    }
    return true;
}

template<std::size_t MaximumSensitivities>
__device__ __forceinline__ void reconstruct_axes(
    const AxisValues<MaximumSensitivities>& axes,
    pg::DeviceSensitivityGraph graph,
    std::size_t row,
    pg::SensitivityOutputs outputs
) {
    for (std::size_t selected = 0U;
         selected < graph.first_count;
         ++selected) {
        const auto coordinate = graph.first[selected];
        outputs.gradients[row * graph.first_count + selected] =
            pg::reconstruct_first_sensitivity(
                axes.stencils[coordinate], axes.values[coordinate]
            );
    }
    for (std::size_t selected = 0U;
         selected < graph.diagonal_second_count;
         ++selected) {
        const auto coordinate = graph.diagonal_second[selected];
        outputs.diagonal_hessians[
            row * graph.diagonal_second_count + selected
        ] = pg::reconstruct_second_sensitivity(
            axes.stencils[coordinate], axes.values[coordinate]
        );
    }
}

template<
    typename Preparation,
    std::size_t MaximumSensitivities,
    typename Inputs,
    typename Policy>
__device__ __forceinline__ bool prepare_and_evaluate_mixed(
    Inputs inputs,
    mcpg::DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph graph,
    const typename Preparation::Scenario& central,
    float central_value,
    const AxisValues<MaximumSensitivities>& axes,
    std::size_t row,
    preparation::Error* preparation_error,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs,
    pg::MixedSensitivityOutputs mixed_outputs
) {
    for (std::size_t selected = 0U;
         selected < graph.mixed_second_count;
         ++selected) {
        const auto pair = graph.mixed_second[selected];
        const auto stencil = pg::make_mixed_sensitivity_stencil(
            axes.stencils[pair.first], axes.stencils[pair.second]
        );
        pg::MixedSensitivityValues values{};
        for (std::size_t node = 0U; node < stencil.node_count; ++node) {
            const auto first_local = stencil.first_local_nodes[node];
            const auto second_local = stencil.second_local_nodes[node];
            if (first_local == 0U && second_local == 0U) {
                values[node] = central_value;
            } else if (first_local == 0U) {
                values[node] = axes.values[pair.second][second_local];
            } else if (second_local == 0U) {
                values[node] = axes.values[pair.first][first_local];
            } else {
                typename Preparation::Scenario corner{};
                if (!pg::prepare_mixed_corner_from_stencils<Preparation>(
                        central,
                        inputs.sensitivities[pair.first],
                        axes.stencils[pair.first],
                        first_local,
                        inputs.sensitivities[pair.second],
                        axes.stencils[pair.second],
                        second_local,
                        plan.time,
                        corner
                    )) {
                    preparation::record_error(
                        preparation_error,
                        preparation::no_admissible_stencil,
                        row,
                        pair.first
                    );
                    return false;
                }
                values[node] = Policy::evaluate(corner);
            }
        }
        const auto output = row * graph.mixed_second_count + selected;
        mixed_stencil_outputs.stencils[output] = stencil;
        mixed_outputs.hessians[output] = pg::reconstruct_mixed_sensitivity(
            stencil, values, central_value
        );
    }
    return true;
}

}  // namespace mixed_detail

template<
    typename Policy,
    typename Preparation,
    typename Inputs,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities>
__global__ void device_prepared_mixed_kernel(
    Inputs inputs,
    mcpg::DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph graph,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs
) {
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t local_row =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         local_row < launch.result_count;
         local_row += stride) {
        const std::size_t row = launch.result_offset + local_row;
        typename Preparation::Scenario central{};
        if (!inputs.make_central(row, plan, central)) {
            preparation::record_error(
                stencil_outputs.error,
                preparation::invalid_central,
                row,
                0U
            );
            continue;
        }
        const float central_value = Policy::evaluate(central);
        outputs.prices[row] = central_value;

        mixed_detail::AxisValues<MaximumSensitivities> axes{};
        if (!mixed_detail::prepare_and_evaluate_axes<
                Preparation,
                MaximumSensitivities,
                Inputs,
                Policy
            >(
                inputs,
                plan,
                graph,
                central,
                central_value,
                row,
                stencil_outputs,
                axes
            )) {
            continue;
        }
        mixed_detail::reconstruct_axes(axes, graph, row, outputs);
        mixed_detail::prepare_and_evaluate_mixed<
            Preparation,
            MaximumSensitivities,
            Inputs,
            Policy
        >(
            inputs,
            plan,
            graph,
            central,
            central_value,
            axes,
            row,
            stencil_outputs.error,
            mixed_stencil_outputs,
            mixed_outputs
        );
    }
}

template<
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    typename HostPlan>
void validate_device_prepared_mixed_launch(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    typename HostPlan::MixedStencilOutputs mixed_stencil_outputs,
    pg::DeviceSensitivityGraph graph,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs
) {
    if (launch.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "Analytical mixed sensitivities require closed-form pricing."
        );
    }
    if (host.sensitivity_count() == 0U
        || host.sensitivity_count() > MaximumSensitivities
        || host.sensitivity_graph.mixed_second.empty()
        || host.sensitivity_graph.mixed_second.size()
            > MaximumMixedSensitivities) {
        throw std::invalid_argument(
            "Analytical mixed sensitivity dimensions exceed their binding."
        );
    }
    mcpg::validate_device_sensitivity_graph(
        graph, host.sensitivity_graph, host.sensitivity_count()
    );

    const auto rows = host.result_count;
    const auto first_count = host.sensitivity_graph.first.size();
    const auto diagonal_count =
        host.sensitivity_graph.diagonal_second.size();
    const auto mixed_count = host.sensitivity_graph.mixed_second.size();
    if (outputs.prices == nullptr || outputs.price_capacity < rows
        || (first_count != 0U
            && (outputs.gradients == nullptr
                || outputs.sensitivity_capacity < rows * first_count))
        || (diagonal_count != 0U
            && (outputs.diagonal_hessians == nullptr
                || outputs.sensitivity_capacity < rows * diagonal_count))
        || mixed_outputs.hessians == nullptr
        || mixed_outputs.capacity < rows * mixed_count
        || mixed_stencil_outputs.stencils == nullptr
        || mixed_stencil_outputs.capacity < rows * mixed_count) {
        throw std::invalid_argument(
            "Insufficient analytical mixed output capacity."
        );
    }

    std::vector<pg::BufferRange> ranges{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
        pg::checked_buffer_range(
            mixed_outputs.hessians, rows * mixed_count, sizeof(float)
        ),
        pg::checked_buffer_range(
            mixed_stencil_outputs.stencils,
            rows * mixed_count,
            sizeof(pg::MixedSensitivityStencil)
        ),
    };
    if (first_count != 0U) {
        ranges.push_back(pg::checked_buffer_range(
            outputs.gradients, rows * first_count, sizeof(float)
        ));
    }
    if (diagonal_count != 0U) {
        ranges.push_back(pg::checked_buffer_range(
            outputs.diagonal_hessians,
            rows * diagonal_count,
            sizeof(float)
        ));
    }
    mcpg::validate_device_prepared_launch(
        device,
        mcpg::make_device_prepared_plan(host),
        stencil_outputs,
        launch,
        ranges
    );
}

template<
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    typename HostPlan>
pg::DeviceSensitivityGraph prepare_device_mixed_launch(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    typename HostPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace_storage,
    std::size_t workspace_bytes
) {
    const auto layout = mixed_workspace_layout(host.sensitivity_graph);
    const auto workspace = make_mixed_workspace(
        workspace_storage, workspace_bytes, layout
    );
    validate_mixed_workspace(workspace, layout);
    const auto graph = pg::upload_sensitivity_graph(
        workspace, host.sensitivity_graph
    );
    validate_device_prepared_mixed_launch<
        MaximumSensitivities,
        MaximumMixedSensitivities
    >(
        host,
        device,
        stencil_outputs,
        mixed_stencil_outputs,
        graph,
        configuration,
        outputs,
        mixed_outputs
    );
    return graph;
}

template<
    typename Policy,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    typename HostPlan>
void launch_device_prepared_mixed(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::DiagonalStencilOutputs stencil_outputs,
    typename HostPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace_storage,
    std::size_t workspace_bytes,
    const char* kernel_name,
    const char* variant
) {
    const auto graph = prepare_device_mixed_launch<
        MaximumSensitivities,
        MaximumMixedSensitivities
    >(
        host,
        device,
        stencil_outputs,
        mixed_stencil_outputs,
        configuration,
        outputs,
        mixed_outputs,
        workspace_storage,
        workspace_bytes
    );

    const auto function = device_prepared_mixed_kernel<
        Policy,
        typename HostPlan::Preparation,
        typename HostPlan::DeviceInputs,
        MaximumSensitivities,
        MaximumMixedSensitivities
    >;
    const std::string diagnostic_variant = std::string(variant)
        + "/K=" + std::to_string(host.sensitivity_count())
        + "/mixed="
        + std::to_string(host.sensitivity_graph.mixed_second.size());
    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        diagnostic_variant.c_str(),
        function,
        dim3(static_cast<unsigned int>(configuration.block_count)),
        dim3(configuration.threads_per_block),
        0U
    );
    function<<<configuration.block_count, configuration.threads_per_block>>>(
        device,
        mcpg::make_device_prepared_plan(host),
        graph,
        configuration,
        outputs,
        mixed_outputs,
        stencil_outputs,
        mixed_stencil_outputs
    );
    check_cuda(cudaGetLastError(), kernel_name);
}

}  // namespace ai_factory::workbench::closed_form::price_gradients
