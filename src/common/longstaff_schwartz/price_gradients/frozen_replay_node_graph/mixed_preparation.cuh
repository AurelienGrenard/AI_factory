// Shared row-local stencil preparation for mixed frozen replay.
#pragma once

#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/value_policy.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_row_preparation.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

template<
    typename Preparation,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    typename Row>
__device__ __forceinline__ bool prepare_frozen_replay_mixed_row(
    const Row& row,
    bool preparation_allowed,
    bool report_preparation_error,
    pg::DeviceSensitivityGraph sensitivity_graph,
    mcpg::MixedNodeGraphWorkspace<FrozenReplayValueNodePolicy> workspace,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs,
    typename Preparation::Scenario* scenarios,
    pg::SensitivityStencil<4U>* stencils,
    mcpg::SensitivityNodeIndices<4U>* axis_node_indices,
    pg::MixedSensitivityStencil* mixed_stencils,
    mcpg::MixedSensitivityNodeIndices* mixed_node_indices,
    std::size_t local_row,
    std::uint16_t& node_count
) {
    int error = preparation::valid;
    std::size_t error_sensitivity = 0U;
    std::uint32_t maximum_steps = 0U;
    bool valid = preparation_allowed
        && row.sensitivity_count > 0U
        && row.sensitivity_count <= MaximumSensitivities;
    if (valid) {
        valid = mcpg::node_graph_detail::
            prepare_mixed_sensitivity_row_from_central<
                Preparation,
                MaximumSensitivities,
                MaximumMixedSensitivities
            >(
                row.central,
                row.sensitivities,
                row.sensitivity_count,
                sensitivity_graph,
                row.time,
                scenarios,
                stencils,
                axis_node_indices,
                mixed_stencils,
                mixed_node_indices,
                node_count,
                maximum_steps,
                error,
                error_sensitivity
            );
    } else if (report_preparation_error) {
        error = preparation::unsupported_order;
    }

    workspace.row_status[local_row] = static_cast<std::uint8_t>(valid);
    if (valid) {
        for (std::size_t sensitivity = 0U;
             sensitivity < row.sensitivity_count;
             ++sensitivity) {
            row.represented_stencils[
                row.result_index * row.sensitivity_count + sensitivity
            ] = stencils[sensitivity];
            workspace.axis_node_indices[
                local_row * row.sensitivity_count + sensitivity
            ] = axis_node_indices[sensitivity];
        }
        for (std::size_t pair = 0U;
             pair < sensitivity_graph.mixed_second_count;
             ++pair) {
            mixed_stencil_outputs.stencils[
                row.result_index * sensitivity_graph.mixed_second_count
                    + pair
            ] = mixed_stencils[pair];
            workspace.mixed_node_indices[
                local_row * sensitivity_graph.mixed_second_count + pair
            ] = mixed_node_indices[pair];
        }
    } else if (report_preparation_error) {
        preparation::record_error(
            row.preparation_error,
            error,
            row.result_index,
            error_sensitivity
        );
    }
    return valid;
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
