// Replay, reconstruct and finalize selected mixed frozen-replay derivatives.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/mixed_reconstruction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;

template<
    typename Policy,
    typename PreparationKernel,
    typename EvaluationKernel>
std::size_t finish_frozen_replay_mixed_node_graph_batch(
    const typename Policy::DeviceInputs& inputs,
    const typename Policy::PreparedRow* rows,
    std::size_t batch_row_count,
    unsigned int reduction_threads,
    std::size_t paths_per_row,
    const longstaff_schwartz::RegressionDiagnostics* diagnostics,
    PreparationKernel prepare,
    unsigned int preparation_threads,
    std::size_t preparation_shared_bytes,
    EvaluationKernel evaluate,
    unsigned int evaluation_threads,
    std::size_t evaluation_shared_bytes,
    const char* name
) {
    const auto graph_configuration = inputs.node_graph_configuration;
    const auto sensitivity_graph = inputs.sensitivity_graph;
    const auto prepared_workspace = inputs.node_graph_workspace;
    const auto workspace = prepared_workspace.graph;
    const auto output_count = frozen_replay_mixed_output_count(sensitivity_graph);
    if (output_count == 0U) return 0U;

    const auto reduction_shared =
        2U * (reduction_threads / 32U) * sizeof(double);
    std::size_t launch_count = 0U;
    const std::string preparation_name =
        std::string(name) + ".frozen_replay_mixed_row_preparation";
    const std::string evaluation_name =
        std::string(name) + ".frozen_replay_mixed_node_evaluation";
    const std::string accumulation_name =
        std::string(name) + ".frozen_replay_mixed_node_moments";
    const std::string finalization_name =
        std::string(name) + ".finalize_frozen_replay_mixed_sensitivities";

    for (std::size_t row_offset = 0U;
         row_offset < batch_row_count;
         row_offset += graph_configuration.row_chunk_size) {
        const auto row_count = std::min(
            graph_configuration.row_chunk_size,
            batch_row_count - row_offset
        );
        const auto* chunk_rows = rows + row_offset;
        const auto* chunk_diagnostics = diagnostics + row_offset;
        const dim3 preparation_grid(
            static_cast<unsigned int>(row_count)
        );
        report_cuda_kernel_phase_launch_if_enabled(
            preparation_name.c_str(),
            "frozen_replay",
            "mixed_row_preparation",
            prepare,
            preparation_grid,
            dim3(preparation_threads),
            preparation_shared_bytes
        );
        prepare<<<
            preparation_grid,
            preparation_threads,
            preparation_shared_bytes
        >>>(
            chunk_rows,
            row_count,
            chunk_diagnostics,
            sensitivity_graph,
            prepared_workspace,
            inputs.stencil_outputs,
            inputs.mixed_stencil_outputs
        );
        check_cuda(
            cudaGetLastError(),
            "frozen-replay mixed row preparation"
        );
        ++launch_count;

        for (std::size_t first_path = 0U;
             first_path < paths_per_row;
             first_path += graph_configuration.path_chunk_size) {
            const auto path_count = std::min(
                graph_configuration.path_chunk_size,
                paths_per_row - first_path
            );
            const dim3 evaluation_grid(
                static_cast<unsigned int>(row_count),
                graph_configuration.path_shards
            );
            report_cuda_kernel_phase_launch_if_enabled(
                evaluation_name.c_str(),
                "frozen_replay",
                "mixed_node_evaluation",
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
                graph_configuration.path_chunk_size,
                chunk_diagnostics,
                sensitivity_graph,
                prepared_workspace,
                inputs.stencil_outputs,
                inputs.mixed_stencil_outputs
            );
            check_cuda(
                cudaGetLastError(),
                "frozen-replay mixed node evaluation"
            );
            ++launch_count;

            const auto accumulate =
                accumulate_frozen_replay_mixed_node_moments_kernel<Policy>;
            const dim3 accumulation_grid(
                static_cast<unsigned int>(row_count),
                static_cast<unsigned int>(output_count)
            );
            report_cuda_kernel_phase_launch_if_enabled(
                accumulation_name.c_str(),
                "frozen_replay",
                "mixed_moment_accumulation",
                accumulate,
                accumulation_grid,
                dim3(reduction_threads),
                0U
            );
            accumulate<<<accumulation_grid, reduction_threads>>>(
                chunk_rows,
                row_count,
                first_path,
                path_count,
                graph_configuration.path_chunk_size,
                inputs.sensitivity_count,
                sensitivity_graph,
                workspace,
                inputs.stencil_outputs,
                inputs.mixed_stencil_outputs
            );
            check_cuda(
                cudaGetLastError(),
                "frozen-replay mixed moment accumulation"
            );
            ++launch_count;
        }

        const auto finalize =
            finalize_frozen_replay_mixed_node_moments_kernel<Policy>;
        const dim3 finalization_grid(
            static_cast<unsigned int>(row_count),
            static_cast<unsigned int>(output_count)
        );
        report_cuda_kernel_phase_launch_if_enabled(
            finalization_name.c_str(),
            "frozen_replay",
            "mixed_finalization",
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
            paths_per_row,
            chunk_diagnostics,
            sensitivity_graph,
            workspace,
            inputs.outputs,
            inputs.mixed_outputs
        );
        check_cuda(
            cudaGetLastError(),
            "finalize frozen-replay mixed sensitivities"
        );
        ++launch_count;
    }
    return launch_count;
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
