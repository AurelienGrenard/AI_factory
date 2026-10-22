// Reconstruct frozen-replay sensitivities from pathwise graph-node values.
#pragma once

#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/longstaff_schwartz/regression_status.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/value_policy.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/price_gradients/reconstruction.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;

template<pg::SensitivityOrders Orders>
__host__ __device__ constexpr std::size_t frozen_replay_node_graph_output_count(
    std::size_t sensitivity_count
) {
    static_assert(pg::requests_second_v<Orders>);
    constexpr std::size_t channels =
        (pg::requests_first_v<Orders> ? 1U : 0U) + 1U;
    return channels * sensitivity_count;
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
__device__ __forceinline__ float reconstruct_frozen_replay_sample(
    const mcpg::TerminalNodeGraphWorkspace<
        FrozenReplayValueNodePolicy
    >& workspace,
    const mcpg::DevicePreparedStencilOutputs<4U>& stencil_outputs,
    std::size_t local_row,
    std::size_t result_index,
    std::size_t local_path,
    std::size_t path_capacity,
    std::size_t sensitivity_count,
    std::size_t output
) {
    const std::size_t node_capacity =
        mcpg::terminal_node_graph_active_node_capacity(sensitivity_count);
    const auto* node_values = workspace.node_values
        + (local_row * path_capacity + local_path) * node_capacity;

    bool first = false;
    std::size_t sensitivity = output;
    if constexpr (pg::requests_first_v<Orders>) {
        first = output < sensitivity_count;
        sensitivity = first ? output : output - sensitivity_count;
    }

    const auto& stencil = stencil_outputs.stencils[
        result_index * sensitivity_count + sensitivity
    ];
    const auto& indices = workspace.node_indices[
        local_row * sensitivity_count + sensitivity
    ];
    if constexpr (pg::requests_first_v<Orders>) {
        if (first && stencil.kind == pg::StencilKind::centered) {
            return (node_values[indices[2U]] - node_values[indices[1U]])
                / stencil.represented_width;
        }
    }

    pg::SensitivityValues<4U> represented{};
    const auto active = pg::active_node_count(stencil);
    for (std::size_t local_node = 0U;
         local_node < active;
         ++local_node) {
        represented[local_node] = node_values[indices[local_node]];
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
    typename Policy,
    std::size_t MaximumSensitivities>
__global__ void accumulate_frozen_replay_node_moments_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t paths_per_row,
    std::size_t path_capacity,
    std::size_t sensitivity_count,
    mcpg::TerminalNodeGraphWorkspace<FrozenReplayValueNodePolicy> workspace,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    double* partials
) {
    const auto output_count =
        frozen_replay_node_graph_output_count<Orders>(sensitivity_count);
    const std::size_t task = blockIdx.y;
    const std::size_t local_row = task / output_count;
    const std::size_t output = task % output_count;
    if (local_row >= row_count
        || workspace.row_status[local_row] == 0U) {
        return;
    }

    double sum = 0.0;
    double square = 0.0;
    const auto result_index = rows[local_row].result_index;
    const std::size_t end_path = first_path + path_count;
    for (std::size_t path =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x
                 + threadIdx.x;
         path < paths_per_row;
         path += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        if (path < first_path || path >= end_path) continue;
        const float sample = reconstruct_frozen_replay_sample<
            Orders, MaximumSensitivities
        >(
            workspace,
            stencil_outputs,
            local_row,
            result_index,
            path - first_path,
            path_capacity,
            sensitivity_count,
            output
        );
        const double value = static_cast<double>(sample);
        sum += value;
        square += value * value;
    }
    const auto total = reductions::reduce_block(sum, square);
    if (threadIdx.x != 0U) return;
    const auto base = (task * 2U) * gridDim.x + blockIdx.x;
    if (first_path == 0U) {
        partials[base] = total.sum;
        partials[base + gridDim.x] = total.sumsq;
    } else {
        partials[base] += total.sum;
        partials[base + gridDim.x] += total.sumsq;
    }
}

template<pg::SensitivityOrders Orders, typename Policy>
__global__ void finalize_frozen_replay_node_moments_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    std::size_t sensitivity_count,
    std::size_t paths_per_row,
    std::size_t blocks_per_row,
    const longstaff_schwartz::RegressionDiagnostics* diagnostics,
    mcpg::TerminalNodeGraphWorkspace<FrozenReplayValueNodePolicy> workspace,
    const double* partials,
    pg::SensitivityOutputs outputs
) {
    const auto output_count =
        frozen_replay_node_graph_output_count<Orders>(sensitivity_count);
    const std::size_t task = blockIdx.x;
    const std::size_t local_row = task / output_count;
    const std::size_t output = task % output_count;
    if (local_row >= row_count) return;

    const auto& row = rows[local_row];
    const auto invalidate = [&] {
        if (threadIdx.x != 0U) return;
        bool first = false;
        std::size_t sensitivity = output;
        if constexpr (pg::requests_first_v<Orders>) {
            first = output < sensitivity_count;
            sensitivity = first ? output : output - sensitivity_count;
        }
        const auto index = row.result_index * sensitivity_count + sensitivity;
        if constexpr (pg::requests_first_v<Orders>) {
            if (first) {
                outputs.gradients[index] = nanf("");
                outputs.gradient_standard_errors[index] = nanf("");
                return;
            }
        }
        outputs.diagonal_hessians[index] = nanf("");
        outputs.diagonal_hessian_standard_errors[index] = nanf("");
    };
    if (workspace.row_status[local_row] == 0U
        || diagnostics[local_row].fatal_failure_count != 0U
        || Policy::initial_exercise_decision(row)
            == longstaff_schwartz::InitialExerciseDecision::invalid) {
        invalidate();
        return;
    }

    double sum = 0.0;
    double square = 0.0;
    const auto base = task * 2U * blocks_per_row;
    for (std::size_t block = threadIdx.x;
         block < blocks_per_row;
         block += blockDim.x) {
        sum += partials[base + block];
        square += partials[base + blocks_per_row + block];
    }
    const auto total = reductions::reduce_block(sum, square);
    if (threadIdx.x != 0U) return;

    double value = 0.0;
    double error = 0.0;
    reductions::compute_statistics(
        total,
        paths_per_row,
        value,
        error
    );
    if constexpr (Policy::kCanExerciseAtInitialTime
                  && !Policy::kFrozenRegressionPolicy) {
        if (Policy::initial_exercise_decision(row)
            == longstaff_schwartz::InitialExerciseDecision::exercise) {
            error = 0.0;
        }
    }

    bool first = false;
    std::size_t sensitivity = output;
    if constexpr (pg::requests_first_v<Orders>) {
        first = output < sensitivity_count;
        sensitivity = first ? output : output - sensitivity_count;
    }
    const auto index = row.result_index * sensitivity_count + sensitivity;
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

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
