// Pathwise reconstruction and moments for selected mixed terminal outputs.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil_outputs.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/mixed_sensitivity_reconstruction.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

template<typename NodePolicy>
__device__ __forceinline__ float reconstruct_mixed_graph_sample(
    const MixedNodeGraphWorkspace<NodePolicy>& workspace,
    const DevicePreparedStencilOutputs<4U>& stencil_outputs,
    const pg::MixedSensitivityStencilOutputs& mixed_stencil_outputs,
    pg::DeviceSensitivityGraph graph,
    std::size_t sensitivity_count,
    std::size_t local_row,
    std::size_t row,
    std::size_t local_path,
    std::size_t path_capacity,
    std::size_t output
) {
    const auto* values = workspace.node_values
        + (local_row * path_capacity + local_path) * graph.node_capacity;
    const auto* metadata = workspace.node_metadata
        + local_row * graph.node_capacity;
    if (output == 0U) {
        return NodePolicy::payoff(metadata[0U], values[0U]);
    }

    --output;
    if (output < graph.first_count) {
        const auto sensitivity = graph.first[output];
        const auto& stencil = stencil_outputs.stencils[
            row * sensitivity_count + sensitivity
        ];
        const auto& indices = workspace.axis_node_indices[
            local_row * sensitivity_count + sensitivity
        ];
        if (stencil.kind == pg::StencilKind::centered) {
            return NodePolicy::centered_first(
                metadata[indices[1U]],
                metadata[indices[2U]],
                values[indices[1U]],
                values[indices[2U]],
                stencil.represented_width
            );
        }
        pg::SensitivityValues<4U> represented{};
        const auto active = pg::active_node_count(stencil);
        for (std::size_t local = 0U; local < active; ++local) {
            const auto node = indices[local];
            represented[local] =
                NodePolicy::payoff(metadata[node], values[node]);
        }
        return pg::reconstruct_first_sensitivity(stencil, represented);
    }

    output -= graph.first_count;
    if (output < graph.diagonal_second_count) {
        const auto sensitivity = graph.diagonal_second[output];
        const auto& stencil = stencil_outputs.stencils[
            row * sensitivity_count + sensitivity
        ];
        const auto& indices = workspace.axis_node_indices[
            local_row * sensitivity_count + sensitivity
        ];
        pg::SensitivityValues<4U> represented{};
        const auto active = pg::active_node_count(stencil);
        for (std::size_t local = 0U; local < active; ++local) {
            const auto node = indices[local];
            represented[local] =
                NodePolicy::payoff(metadata[node], values[node]);
        }
        return pg::reconstruct_second_sensitivity(stencil, represented);
    }

    const auto pair = output - graph.diagonal_second_count;
    const auto& stencil = mixed_stencil_outputs.stencils[
        row * graph.mixed_second_count + pair
    ];
    const auto& indices = workspace.mixed_node_indices[
        local_row * graph.mixed_second_count + pair
    ];
    pg::MixedSensitivityValues represented{};
    for (std::size_t local = 0U; local < stencil.node_count; ++local) {
        const auto node = indices[local];
        represented[local] = NodePolicy::payoff(
            metadata[node], values[node]
        );
    }
    const float central = NodePolicy::payoff(metadata[0U], values[0U]);
    return pg::reconstruct_mixed_sensitivity(
        stencil, represented, central
    );
}

template<typename NodePolicy>
__global__ void accumulate_mixed_node_moments_kernel(
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    std::size_t sensitivity_count,
    pg::DeviceSensitivityGraph graph,
    MixedNodeGraphWorkspace<NodePolicy> workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs
) {
    const std::size_t local_row = blockIdx.x;
    const std::size_t output = blockIdx.y;
    if (local_row >= row_count || output >= graph.output_count()
        || workspace.row_status[local_row] == 0U) {
        return;
    }
    const auto moment_index =
        (local_row * graph.output_count() + output) * blockDim.x + threadIdx.x;
    reductions::MomentSums moments = first_path == 0U
        ? reductions::MomentSums{0.0, 0.0}
        : workspace.thread_moments[moment_index];
    const auto row = first_row + local_row;
    for (std::size_t path = first_path + threadIdx.x;
         path < first_path + path_count;
         path += blockDim.x) {
        const float sample = reconstruct_mixed_graph_sample(
            workspace,
            stencil_outputs,
            mixed_stencil_outputs,
            graph,
            sensitivity_count,
            local_row,
            row,
            path - first_path,
            path_capacity,
            output
        );
        const double value = static_cast<double>(sample);
        moments.sum += value;
        moments.sumsq += value * value;
    }
    workspace.thread_moments[moment_index] = moments;
}

template<typename NodePolicy>
__global__ void finalize_mixed_node_moments_kernel(
    std::size_t first_row,
    std::size_t row_count,
    std::size_t paths_per_row,
    pg::DeviceSensitivityGraph graph,
    MixedNodeGraphWorkspace<NodePolicy> workspace,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs
) {
    const std::size_t local_row = blockIdx.x;
    const std::size_t output = blockIdx.y;
    if (local_row >= row_count || output >= graph.output_count()
        || workspace.row_status[local_row] == 0U) {
        return;
    }
    const auto moment_index =
        (local_row * graph.output_count() + output) * blockDim.x + threadIdx.x;
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
    const auto row = first_row + local_row;
    if (output == 0U) {
        outputs.prices[row] = static_cast<float>(value);
        outputs.price_standard_errors[row] = static_cast<float>(error);
        return;
    }

    auto selected = output - 1U;
    if (selected < graph.first_count) {
        const auto index = row * graph.first_count + selected;
        outputs.gradients[index] = static_cast<float>(value);
        outputs.gradient_standard_errors[index] = static_cast<float>(error);
        return;
    }
    selected -= graph.first_count;
    if (selected < graph.diagonal_second_count) {
        const auto index = row * graph.diagonal_second_count + selected;
        outputs.diagonal_hessians[index] = static_cast<float>(value);
        outputs.diagonal_hessian_standard_errors[index] =
            static_cast<float>(error);
        return;
    }
    selected -= graph.diagonal_second_count;
    const auto index = row * graph.mixed_second_count + selected;
    mixed_outputs.hessians[index] = static_cast<float>(value);
    mixed_outputs.standard_errors[index] = static_cast<float>(error);
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
