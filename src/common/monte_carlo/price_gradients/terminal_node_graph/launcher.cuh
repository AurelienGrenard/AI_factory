// Host orchestration of the bounded three-phase terminal node graph.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/evaluation.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/reconstruction.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = tuning::DefaultTerminalNodeTuning,
    typename Inputs>
void launch_device_prepared_terminal_node_graph(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::LaunchConfiguration& launch,
    TerminalNodeGraphConfiguration graph,
    TerminalNodeGraphWorkspace<
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>
    > workspace,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
    static_assert(pg::requests_second_v<Orders>);
    static_assert(tuning::valid_profile_v<Tuning>);
    constexpr auto node_capacity =
        terminal_node_graph_node_capacity<MaximumSensitivities>();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);

    if (launch.result_count == 0U
        || launch.paths_per_price < 2U
        || plan.sensitivity_count == 0U
        || plan.sensitivity_count > MaximumSensitivities) {
        throw std::invalid_argument(
            "Invalid terminal node-graph dimensions."
        );
    }
    if (launch.threads_per_block < 32U
        || launch.threads_per_block > 1024U
        || launch.threads_per_block % 32U != 0U) {
        throw std::invalid_argument(
            "Terminal node-graph reduction threads must contain whole warps."
        );
    }
    const auto requirements = terminal_node_graph_workspace_requirements<
        Orders, MaximumSensitivities
    >(plan.sensitivity_count, launch.threads_per_block, graph);
    validate_terminal_node_graph_workspace(workspace, requirements);

    using NodePolicy =
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
    using EvaluationKernel = void (*)(
        Inputs,
        DevicePreparedPlan,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        TerminalNodeGraphWorkspace<NodePolicy>,
        DevicePreparedStencilOutputs<4U>,
        std::uint64_t
    );
    EvaluationKernel evaluate =
        node_graph_detail::evaluate_nodes_kernel<
            Orders,
            Dynamics,
            ProductPolicy,
            Preparation,
            MaximumSensitivities,
            GroupSize,
            NodesPerWorker,
            Tuning,
            Inputs
        >;
    if constexpr (Tuning::kLaunchBoundsEnabled) {
        evaluate = node_graph_detail::bounded_evaluate_nodes_kernel<
            Orders,
            Dynamics,
            ProductPolicy,
            Preparation,
            MaximumSensitivities,
            GroupSize,
            NodesPerWorker,
            Tuning,
            Inputs
        >;
    }

    constexpr unsigned int evaluation_threads =
        Tuning::kThreadsPerBlock;
    constexpr unsigned int groups_per_block =
        evaluation_threads / GroupSize;
    constexpr std::size_t scratch_bytes_per_group =
        node_graph_detail::distributed_node_scratch_bytes_v<
            node_capacity, Dynamics
        >;
    const std::size_t evaluation_shared =
        groups_per_block * scratch_bytes_per_group;
    const auto output_count =
        terminal_node_graph_output_count<Orders>(plan.sensitivity_count);
    const auto reduction_shared =
        2U * (launch.threads_per_block / 32U) * sizeof(double);

    for (std::size_t row_offset = 0U;
         row_offset < launch.result_count;
         row_offset += graph.row_chunk_size) {
        const auto row_count = std::min(
            graph.row_chunk_size,
            launch.result_count - row_offset
        );
        const auto first_row = launch.result_offset + row_offset;
        for (std::size_t first_path = 0U;
             first_path < launch.paths_per_price;
             first_path += graph.path_chunk_size) {
            const auto path_count = std::min(
                graph.path_chunk_size,
                launch.paths_per_price - first_path
            );
            const dim3 evaluation_grid(
                static_cast<unsigned int>(row_count),
                graph.path_shards
            );
            report_cuda_kernel_phase_launch_if_enabled(
                kernel_name,
                variant,
                "node_evaluation",
                evaluate,
                evaluation_grid,
                dim3(evaluation_threads),
                evaluation_shared
            );
            evaluate<<<
                evaluation_grid,
                evaluation_threads,
                evaluation_shared
            >>>(
                inputs,
                plan,
                first_row,
                row_count,
                first_path,
                path_count,
                graph.path_chunk_size,
                workspace,
                stencil_outputs,
                launch.base_seed
            );
            check_cuda(
                cudaGetLastError(),
                "terminal sensitivity node evaluation"
            );

            const auto accumulate =
                node_graph_detail::accumulate_node_moments_kernel<
                    Orders, NodePolicy, MaximumSensitivities
                >;
            const dim3 reduction_grid(
                static_cast<unsigned int>(row_count),
                static_cast<unsigned int>(output_count)
            );
            report_cuda_kernel_phase_launch_if_enabled(
                kernel_name,
                variant,
                "moment_accumulation",
                accumulate,
                reduction_grid,
                dim3(launch.threads_per_block)
            );
            accumulate<<<
                reduction_grid,
                launch.threads_per_block
            >>>(
                first_row,
                row_count,
                first_path,
                path_count,
                graph.path_chunk_size,
                plan.sensitivity_count,
                workspace,
                stencil_outputs
            );
            check_cuda(
                cudaGetLastError(),
                "terminal sensitivity moment accumulation"
            );
        }

        const auto finalize =
            node_graph_detail::finalize_node_moments_kernel<
                Orders, NodePolicy
            >;
        const dim3 final_grid(
            static_cast<unsigned int>(row_count),
            static_cast<unsigned int>(output_count)
        );
        report_cuda_kernel_phase_launch_if_enabled(
            kernel_name,
            variant,
            "moment_finalization",
            finalize,
            final_grid,
            dim3(launch.threads_per_block),
            reduction_shared
        );
        finalize<<<
            final_grid,
            launch.threads_per_block,
            reduction_shared
        >>>(
            first_row,
            row_count,
            plan.sensitivity_count,
            launch.paths_per_price,
            workspace,
            outputs
        );
        check_cuda(
            cudaGetLastError(),
            "terminal sensitivity moment finalization"
        );
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
