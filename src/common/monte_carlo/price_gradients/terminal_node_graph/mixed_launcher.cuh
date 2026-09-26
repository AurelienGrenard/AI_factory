// Host orchestration of selected mixed terminal node graphs.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/monte_carlo/price_gradients/node_graph/kernel_shared_memory.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_launch_validation.hpp"
#include "common/monte_carlo/price_gradients/terminal_node_graph/mixed_evaluation.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_reconstruction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo::price_gradients {


template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning = tuning::DefaultTerminalNodeTuning,
    typename Inputs>
void launch_device_prepared_terminal_mixed_node_graph(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::SensitivityGraphPlan& host_graph,
    pg::DeviceSensitivityGraph device_graph,
    const pg::LaunchConfiguration& launch,
    TerminalNodeGraphConfiguration graph_configuration,
    MixedNodeGraphWorkspace<
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>
    > workspace,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
    static_assert(tuning::valid_profile_v<Tuning>);
    constexpr auto node_capacity =
        node_graph_detail::mixed_node_graph_node_capacity<
            MaximumSensitivities, MaximumMixedSensitivities
        >();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);

    if (launch.result_count == 0U || launch.paths_per_price < 2U
        || plan.sensitivity_count == 0U
        || plan.sensitivity_count > MaximumSensitivities
        || host_graph.mixed_second.empty()
        || host_graph.mixed_second.size() > MaximumMixedSensitivities
        || host_graph.node_capacity > node_capacity) {
        throw std::invalid_argument(
            "Invalid mixed terminal node-graph dimensions."
        );
    }
    if (launch.threads_per_block < 32U
        || launch.threads_per_block > 1024U
        || launch.threads_per_block % 32U != 0U) {
        throw std::invalid_argument(
            "Mixed terminal reductions require whole warps."
        );
    }
    validate_mixed_node_graph_launch(
        inputs,
        plan,
        host_graph,
        device_graph,
        launch,
        outputs,
        mixed_outputs,
        stencil_outputs,
        mixed_stencil_outputs
    );

    const auto requirements =
        mixed_node_graph_workspace_requirements(
            plan.sensitivity_count,
            host_graph,
            launch.threads_per_block,
            graph_configuration
        );
    validate_mixed_node_graph_workspace(workspace, requirements);

    using NodePolicy =
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
    const auto evaluate = node_graph_detail::evaluate_mixed_nodes_kernel<
        Dynamics,
        ProductPolicy,
        Preparation,
        MaximumSensitivities,
        MaximumMixedSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning,
        Inputs
    >;
    constexpr unsigned int evaluation_threads = Tuning::kThreadsPerBlock;
    constexpr unsigned int groups_per_block =
        evaluation_threads / GroupSize;
    constexpr std::size_t scratch_bytes_per_group =
        node_graph_detail::path_team_scratch_bytes_v<
            node_capacity, Dynamics
        >;
    const auto scenario_shared = (
        host_graph.node_capacity * sizeof(typename Preparation::Scenario)
        + 15U
    ) & ~std::size_t{15U};
    const std::size_t evaluation_shared = scenario_shared
        + groups_per_block * scratch_bytes_per_group;
    configure_node_graph_dynamic_shared_memory(
        evaluate,
        evaluation_shared,
        "mixed terminal evaluation kernel attributes",
        "mixed terminal CUDA device",
        "mixed terminal CUDA device properties",
        "Mixed terminal sensitivity graph exceeds device shared memory.",
        "mixed terminal evaluation dynamic shared memory"
    );
    const auto reduction_shared =
        2U * (launch.threads_per_block / 32U) * sizeof(double);

    for (std::size_t row_offset = 0U;
         row_offset < launch.result_count;
         row_offset += graph_configuration.row_chunk_size) {
        const auto row_count = std::min(
            graph_configuration.row_chunk_size,
            launch.result_count - row_offset
        );
        const auto first_row = launch.result_offset + row_offset;
        for (std::size_t first_path = 0U;
             first_path < launch.paths_per_price;
             first_path += graph_configuration.path_chunk_size) {
            const auto path_count = std::min(
                graph_configuration.path_chunk_size,
                launch.paths_per_price - first_path
            );
            const dim3 evaluation_grid(
                static_cast<unsigned int>(row_count),
                graph_configuration.path_shards
            );
            report_cuda_kernel_phase_launch_if_enabled(
                kernel_name,
                variant,
                "mixed_node_evaluation",
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
                device_graph,
                first_row,
                row_count,
                first_path,
                path_count,
                graph_configuration.path_chunk_size,
                workspace,
                stencil_outputs,
                mixed_stencil_outputs,
                launch.base_seed
            );
            check_cuda(
                cudaGetLastError(),
                "mixed terminal sensitivity node evaluation"
            );

            const auto accumulate =
                node_graph_detail::accumulate_mixed_node_moments_kernel<
                    NodePolicy
                >;
            const dim3 reduction_grid(
                static_cast<unsigned int>(row_count),
                static_cast<unsigned int>(host_graph.output_count())
            );
            accumulate<<<
                reduction_grid,
                launch.threads_per_block
            >>>(
                first_row,
                row_count,
                first_path,
                path_count,
                graph_configuration.path_chunk_size,
                plan.sensitivity_count,
                device_graph,
                workspace,
                stencil_outputs,
                mixed_stencil_outputs
            );
            check_cuda(
                cudaGetLastError(),
                "mixed terminal sensitivity moment accumulation"
            );
        }

        const auto finalize =
            node_graph_detail::finalize_mixed_node_moments_kernel<NodePolicy>;
        const dim3 final_grid(
            static_cast<unsigned int>(row_count),
            static_cast<unsigned int>(host_graph.output_count())
        );
        finalize<<<
            final_grid,
            launch.threads_per_block,
            reduction_shared
        >>>(
            first_row,
            row_count,
            launch.paths_per_price,
            device_graph,
            workspace,
            outputs,
            mixed_outputs
        );
        check_cuda(
            cudaGetLastError(),
            "mixed terminal sensitivity moment finalization"
        );
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
