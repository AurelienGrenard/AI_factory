// Reconstruct selected frozen-policy derivatives from mixed graph-node values.
#pragma once

#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/longstaff_schwartz/regression_status.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_exercise_node_graph/value_policy.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_reconstruction.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;

__host__ __device__ inline std::size_t frozen_mixed_output_count(
    pg::DeviceSensitivityGraph graph
) {
    return graph.first_count + graph.diagonal_second_count
        + graph.mixed_second_count;
}

template<typename Policy>
__global__ void accumulate_frozen_mixed_node_moments_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    std::size_t sensitivity_count,
    pg::DeviceSensitivityGraph graph,
    mcpg::MixedNodeGraphWorkspace<FrozenExerciseValueNodePolicy> workspace,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs
) {
    const std::size_t local_row = blockIdx.x;
    const std::size_t selected_output = blockIdx.y;
    if (local_row >= row_count
        || selected_output >= frozen_mixed_output_count(graph)
        || workspace.row_status[local_row] == 0U) {
        return;
    }

    // Slot zero belongs to the central price, which the main LSM pipeline owns.
    const std::size_t graph_output = selected_output + 1U;
    const auto moment_index =
        (local_row * graph.output_count() + graph_output) * blockDim.x
        + threadIdx.x;
    reductions::MomentSums moments = first_path == 0U
        ? reductions::MomentSums{0.0, 0.0}
        : workspace.thread_moments[moment_index];
    const auto result_index = rows[local_row].result_index;
    for (std::size_t path = first_path + threadIdx.x;
         path < first_path + path_count;
         path += blockDim.x) {
        const float sample = mcpg::node_graph_detail::
            reconstruct_mixed_graph_sample(
                workspace,
                stencil_outputs,
                mixed_stencil_outputs,
                graph,
                sensitivity_count,
                local_row,
                result_index,
                path - first_path,
                path_capacity,
                graph_output
            );
        const double value = static_cast<double>(sample);
        moments.sum += value;
        moments.sumsq += value * value;
    }
    workspace.thread_moments[moment_index] = moments;
}

__device__ __forceinline__ void invalidate_frozen_mixed_output(
    std::size_t result_index,
    std::size_t selected_output,
    pg::DeviceSensitivityGraph graph,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs
) {
    if (selected_output < graph.first_count) {
        const auto index = result_index * graph.first_count + selected_output;
        outputs.gradients[index] = nanf("");
        outputs.gradient_standard_errors[index] = nanf("");
        return;
    }
    selected_output -= graph.first_count;
    if (selected_output < graph.diagonal_second_count) {
        const auto index = result_index * graph.diagonal_second_count
            + selected_output;
        outputs.diagonal_hessians[index] = nanf("");
        outputs.diagonal_hessian_standard_errors[index] = nanf("");
        return;
    }
    selected_output -= graph.diagonal_second_count;
    const auto index = result_index * graph.mixed_second_count
        + selected_output;
    mixed_outputs.hessians[index] = nanf("");
    mixed_outputs.standard_errors[index] = nanf("");
}

__device__ __forceinline__ void publish_frozen_mixed_output(
    std::size_t result_index,
    std::size_t selected_output,
    pg::DeviceSensitivityGraph graph,
    float value,
    float error,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs
) {
    if (selected_output < graph.first_count) {
        const auto index = result_index * graph.first_count + selected_output;
        outputs.gradients[index] = value;
        outputs.gradient_standard_errors[index] = error;
        return;
    }
    selected_output -= graph.first_count;
    if (selected_output < graph.diagonal_second_count) {
        const auto index = result_index * graph.diagonal_second_count
            + selected_output;
        outputs.diagonal_hessians[index] = value;
        outputs.diagonal_hessian_standard_errors[index] = error;
        return;
    }
    selected_output -= graph.diagonal_second_count;
    const auto index = result_index * graph.mixed_second_count
        + selected_output;
    mixed_outputs.hessians[index] = value;
    mixed_outputs.standard_errors[index] = error;
}

template<typename Policy>
__global__ void finalize_frozen_mixed_node_moments_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    std::size_t paths_per_row,
    const longstaff_schwartz::RegressionDiagnostics* diagnostics,
    pg::DeviceSensitivityGraph graph,
    mcpg::MixedNodeGraphWorkspace<FrozenExerciseValueNodePolicy> workspace,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs
) {
    const std::size_t local_row = blockIdx.x;
    const std::size_t selected_output = blockIdx.y;
    if (local_row >= row_count
        || selected_output >= frozen_mixed_output_count(graph)) {
        return;
    }

    const auto& row = rows[local_row];
    if (workspace.row_status[local_row] == 0U
        || diagnostics[local_row].fatal_failure_count != 0U
        || Policy::initial_exercise_decision(row)
            == longstaff_schwartz::InitialExerciseDecision::invalid) {
        if (threadIdx.x == 0U) {
            invalidate_frozen_mixed_output(
                row.result_index,
                selected_output,
                graph,
                outputs,
                mixed_outputs
            );
        }
        return;
    }

    const std::size_t graph_output = selected_output + 1U;
    const auto moment_index =
        (local_row * graph.output_count() + graph_output) * blockDim.x
        + threadIdx.x;
    const auto local = workspace.thread_moments[moment_index];
    const auto total = reductions::reduce_block(local.sum, local.sumsq);
    if (threadIdx.x != 0U) return;

    double value = 0.0;
    double error = 0.0;
    reductions::compute_statistics(
        total,
        paths_per_row,
        value,
        error,
        paths_per_row / blockDim.x
            + (paths_per_row % blockDim.x != 0U)
    );
    if constexpr (Policy::kCanExerciseAtInitialTime) {
        if (Policy::initial_exercise_decision(row)
            == longstaff_schwartz::InitialExerciseDecision::exercise) {
            error = 0.0;
        }
    }
    publish_frozen_mixed_output(
        row.result_index,
        selected_output,
        graph,
        static_cast<float>(value),
        static_cast<float>(error),
        outputs,
        mixed_outputs
    );
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
