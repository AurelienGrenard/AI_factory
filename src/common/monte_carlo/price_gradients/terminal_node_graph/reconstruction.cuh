// Pathwise derivative reconstruction and deterministic moment accumulation.
#pragma once

#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

namespace node_graph_detail {

template<
    pg::SensitivityOrders Orders,
    typename NodePolicy,
    std::size_t MaximumSensitivities>
__device__ __forceinline__ float reconstruct_sample(
    const TerminalNodeGraphWorkspace<NodePolicy>& workspace,
    const DevicePreparedStencilOutputs<4U>& stencil_outputs,
    std::size_t local_row,
    std::size_t row,
    std::size_t local_path,
    std::size_t path_capacity,
    std::size_t sensitivity_count,
    std::size_t output
) {
    constexpr std::size_t node_capacity =
        terminal_node_graph_node_capacity<MaximumSensitivities>();
    const auto* values = workspace.node_values
        + (local_row * path_capacity + local_path) * node_capacity;
    const auto* metadata = workspace.node_metadata
        + local_row * node_capacity;
    if (output == 0U) {
        return NodePolicy::payoff(metadata[0U], values[0U]);
    }

    std::size_t sensitivity = 0U;
    bool first = false;
    if constexpr (pg::requests_first_v<Orders>) {
        first = output <= sensitivity_count;
        sensitivity = first
            ? output - 1U
            : output - 1U - sensitivity_count;
    } else {
        sensitivity = output - 1U;
    }

    const auto& stencil =
        stencil_outputs.stencils[row * sensitivity_count + sensitivity];
    const auto& indices = workspace.node_indices[
        local_row * sensitivity_count + sensitivity
    ];
    if constexpr (pg::requests_first_v<Orders>) {
        if (first && stencil.kind == pg::StencilKind::centered) {
            return NodePolicy::centered_first(
                metadata[indices[1U]],
                metadata[indices[2U]],
                values[indices[1U]],
                values[indices[2U]],
                stencil.represented_width
            );
        }
    }

    pg::SensitivityValues<4U> represented{};
    const auto active = pg::active_node_count(stencil);
    for (std::size_t local_node = 0U;
         local_node < active;
         ++local_node) {
        const auto node = indices[local_node];
        represented[local_node] =
            NodePolicy::payoff(metadata[node], values[node]);
    }
    const auto result = pg::reconstruct_sensitivity<Orders>(
        stencil, represented
    );
    if constexpr (pg::requests_first_v<Orders>) {
        return first ? result.first : result.second;
    }
    return result.second;
}

template<
    pg::SensitivityOrders Orders,
    typename NodePolicy,
    std::size_t MaximumSensitivities>
__global__ void accumulate_node_moments_kernel(
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    std::size_t sensitivity_count,
    TerminalNodeGraphWorkspace<NodePolicy> workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs
) {
    const std::size_t local_row = blockIdx.x;
    const std::size_t output = blockIdx.y;
    const auto output_count =
        terminal_node_graph_output_count<Orders>(sensitivity_count);
    if (local_row >= row_count || output >= output_count
        || workspace.row_status[local_row] == 0U) {
        return;
    }

    const auto moment_index =
        (local_row * output_count + output) * blockDim.x + threadIdx.x;
    reductions::MomentSums moments = first_path == 0U
        ? reductions::MomentSums{0.0, 0.0}
        : workspace.thread_moments[moment_index];
    const auto row = first_row + local_row;
    for (std::size_t path = first_path + threadIdx.x;
         path < first_path + path_count;
         path += blockDim.x) {
        const float sample = reconstruct_sample<
            Orders, NodePolicy, MaximumSensitivities
        >(
            workspace,
            stencil_outputs,
            local_row,
            row,
            path - first_path,
            path_capacity,
            sensitivity_count,
            output
        );
        const double value = static_cast<double>(sample);
        moments.sum += value;
        moments.sumsq += value * value;
    }
    workspace.thread_moments[moment_index] = moments;
}

template<pg::SensitivityOrders Orders, typename NodePolicy>
__global__ void finalize_node_moments_kernel(
    std::size_t first_row,
    std::size_t row_count,
    std::size_t sensitivity_count,
    std::size_t paths_per_row,
    TerminalNodeGraphWorkspace<NodePolicy> workspace,
    pg::SensitivityOutputs outputs
) {
    const std::size_t local_row = blockIdx.x;
    const std::size_t output = blockIdx.y;
    const auto output_count =
        terminal_node_graph_output_count<Orders>(sensitivity_count);
    if (local_row >= row_count || output >= output_count
        || workspace.row_status[local_row] == 0U) {
        return;
    }

    const auto moment_index =
        (local_row * output_count + output) * blockDim.x + threadIdx.x;
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

    std::size_t sensitivity = 0U;
    bool first = false;
    if constexpr (pg::requests_first_v<Orders>) {
        first = output <= sensitivity_count;
        sensitivity = first
            ? output - 1U
            : output - 1U - sensitivity_count;
    } else {
        sensitivity = output - 1U;
    }
    const auto index = row * sensitivity_count + sensitivity;
    if constexpr (pg::requests_first_v<Orders>) {
        if (first) {
            outputs.gradients[index] = static_cast<float>(value);
            outputs.gradient_standard_errors[index] =
                static_cast<float>(error);
            return;
        }
    }
    outputs.diagonal_hessians[index] = static_cast<float>(value);
    outputs.diagonal_hessian_standard_errors[index] =
        static_cast<float>(error);
}

}  // namespace node_graph_detail

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
