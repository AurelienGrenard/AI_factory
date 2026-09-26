// Three-phase replay, reconstruction and finalization of frozen LSM nodes.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_exercise_node_graph/reconstruction.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    std::size_t MaximumSensitivities,
    typename EvaluationKernel>
std::size_t finish_frozen_exercise_node_graph_batch(
    const typename Policy::DeviceInputs& inputs,
    const typename Policy::PreparedRow* rows,
    std::size_t batch_row_count,
    unsigned int reduction_threads,
    std::size_t paths_per_row,
    std::size_t blocks_per_row,
    double* partials,
    const longstaff_schwartz::RegressionDiagnostics* diagnostics,
    EvaluationKernel evaluate,
    unsigned int evaluation_threads,
    std::size_t evaluation_shared_bytes,
    const char* name
) {
    static_assert(pg::requests_second_v<Orders>);
    if (inputs.sensitivity_count == 0U) return 0U;

    const auto graph = inputs.node_graph_configuration;
    const auto workspace = inputs.node_graph_workspace;
    const auto output_count =
        frozen_node_graph_output_count<Orders>(inputs.sensitivity_count);
    const auto reduction_shared =
        2U * (reduction_threads / 32U) * sizeof(double);
    std::size_t launch_count = 0U;

    const std::string evaluation_name =
        std::string(name) + ".frozen_node_evaluation";
    const std::string accumulation_name =
        std::string(name) + ".frozen_node_moments";
    const std::string finalization_name =
        std::string(name) + ".finalize_frozen_node_sensitivities";

    for (std::size_t row_offset = 0U;
         row_offset < batch_row_count;
         row_offset += graph.row_chunk_size) {
        const auto row_count = std::min(
            graph.row_chunk_size,
            batch_row_count - row_offset
        );
        const auto* chunk_rows = rows + row_offset;
        const auto* chunk_diagnostics = diagnostics + row_offset;
        for (std::size_t first_path = 0U;
             first_path < paths_per_row;
             first_path += graph.path_chunk_size) {
            const auto path_count = std::min(
                graph.path_chunk_size,
                paths_per_row - first_path
            );
            const dim3 evaluation_grid(
                static_cast<unsigned int>(row_count),
                graph.path_shards
            );
            report_cuda_kernel_phase_launch_if_enabled(
                evaluation_name.c_str(),
                "frozen_policy",
                "node_evaluation",
                evaluate,
                evaluation_grid,
                dim3(evaluation_threads),
                evaluation_shared_bytes
            );
            evaluate<<<
                evaluation_grid,
                evaluation_threads,
                evaluation_shared_bytes
            >>>(
                chunk_rows,
                row_count,
                first_path,
                path_count,
                graph.path_chunk_size,
                chunk_diagnostics,
                workspace,
                inputs.stencil_outputs
            );
            check_cuda(
                cudaGetLastError(),
                "frozen-exercise sensitivity node evaluation"
            );
            ++launch_count;

            const auto accumulate =
                accumulate_frozen_node_moments_kernel<
                    Orders, Policy, MaximumSensitivities
                >;
            const dim3 accumulation_grid(
                static_cast<unsigned int>(blocks_per_row),
                static_cast<unsigned int>(row_count * output_count)
            );
            report_cuda_kernel_phase_launch_if_enabled(
                accumulation_name.c_str(),
                "frozen_policy",
                "moment_accumulation",
                accumulate,
                accumulation_grid,
                dim3(reduction_threads)
            );
            accumulate<<<accumulation_grid, reduction_threads>>>(
                chunk_rows,
                row_count,
                first_path,
                path_count,
                paths_per_row,
                graph.path_chunk_size,
                inputs.sensitivity_count,
                workspace,
                inputs.stencil_outputs,
                partials
            );
            check_cuda(
                cudaGetLastError(),
                "frozen-exercise node moment accumulation"
            );
            ++launch_count;
        }

        const auto finalize = finalize_frozen_node_moments_kernel<
            Orders, Policy
        >;
        const dim3 finalization_grid(
            static_cast<unsigned int>(row_count * output_count)
        );
        report_cuda_kernel_phase_launch_if_enabled(
            finalization_name.c_str(),
            "frozen_policy",
            "moment_finalization",
            finalize,
            finalization_grid,
            dim3(reduction_threads),
            reduction_shared
        );
        finalize<<<
            finalization_grid,
            reduction_threads,
            reduction_shared
        >>>(
            chunk_rows,
            row_count,
            inputs.sensitivity_count,
            paths_per_row,
            blocks_per_row,
            chunk_diagnostics,
            workspace,
            partials,
            inputs.outputs
        );
        check_cuda(
            cudaGetLastError(),
            "finalize frozen-exercise node sensitivities"
        );
        ++launch_count;
    }
    return launch_count;
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
