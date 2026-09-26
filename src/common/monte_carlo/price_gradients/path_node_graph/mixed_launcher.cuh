// Host orchestration of selected mixed path sensitivity node graphs.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/monte_carlo/price_gradients/node_graph/kernel_shared_memory.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_launch_validation.hpp"
#include "common/monte_carlo/price_gradients/node_graph/mixed_reconstruction.cuh"
#include "common/monte_carlo/price_gradients/path_node_graph/mixed_evaluation.cuh"
#include "common/monte_carlo/price_gradients/path_node_graph/mixed_preparation.cuh"
#include "common/monte_carlo/price_gradients/path_node_graph/mixed_workspace.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"

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
    typename Schedule,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int TeamSize,
    unsigned int NodesPerWorker,
    typename Tuning = tuning::DefaultTerminalNodeTuning,
    typename Inputs>
void launch_device_prepared_path_mixed_node_graph(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::SensitivityGraphPlan& host_graph,
    pg::DeviceSensitivityGraph device_graph,
    const pg::LaunchConfiguration& launch,
    TerminalNodeGraphConfiguration graph_configuration,
    MixedPathNodeGraphWorkspace<
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>,
        Dynamics,
        Schedule
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
    static_assert(TeamSize * NodesPerWorker >= node_capacity);
    static_assert(Tuning::kThreadsPerBlock % TeamSize == 0U);

    if (launch.result_count == 0U || launch.paths_per_price < 2U
        || plan.sensitivity_count == 0U
        || plan.sensitivity_count > MaximumSensitivities
        || host_graph.mixed_second.empty()
        || host_graph.mixed_second.size() > MaximumMixedSensitivities
        || host_graph.node_capacity > node_capacity) {
        throw std::invalid_argument(
            "Invalid mixed path node-graph dimensions."
        );
    }
    if (launch.threads_per_block < 32U
        || launch.threads_per_block > 1024U
        || launch.threads_per_block % 32U != 0U) {
        throw std::invalid_argument(
            "Mixed path reductions require whole warps."
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

    using NodePolicy =
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>;
    const auto layout = mixed_path_node_graph_workspace_layout<
        NodePolicy, Dynamics, Schedule
    >(
        plan.sensitivity_count,
        host_graph,
        launch.threads_per_block,
        graph_configuration
    );
    validate_mixed_path_node_graph_workspace(
        workspace, layout.capacities
    );

    const auto prepare = path_node_graph_detail::prepare_mixed_rows_kernel<
        Dynamics,
        ProductPolicy,
        Preparation,
        Schedule,
        MaximumSensitivities,
        MaximumMixedSensitivities,
        Inputs
    >;
    const auto evaluate = path_node_graph_detail::evaluate_mixed_nodes_kernel<
        Dynamics,
        ProductPolicy,
        Preparation,
        Schedule,
        MaximumSensitivities,
        MaximumMixedSensitivities,
        TeamSize,
        NodesPerWorker,
        Tuning
    >;
    constexpr unsigned int evaluation_threads = Tuning::kThreadsPerBlock;
    constexpr unsigned int teams_per_block =
        evaluation_threads / TeamSize;
    const auto preparation_shared = (
        host_graph.node_capacity * sizeof(typename Preparation::Scenario)
        + 15U
    ) & ~std::size_t{15U};
    constexpr auto scratch_bytes_per_team =
        path_node_graph_detail::mixed_path_team_scratch_bytes_v<
            node_capacity, Dynamics
        >;
    const std::size_t evaluation_shared =
        teams_per_block * scratch_bytes_per_team;
    configure_node_graph_dynamic_shared_memory(
        prepare,
        preparation_shared,
        "mixed path preparation kernel attributes",
        "mixed path preparation CUDA device",
        "mixed path preparation CUDA device properties",
        "Mixed path preparation exceeds device shared memory.",
        "mixed path preparation dynamic shared memory"
    );
    configure_node_graph_dynamic_shared_memory(
        evaluate,
        evaluation_shared,
        "mixed path evaluation kernel attributes",
        "mixed path evaluation CUDA device",
        "mixed path evaluation CUDA device properties",
        "Mixed path evaluation exceeds device shared memory.",
        "mixed path evaluation dynamic shared memory"
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
        const dim3 preparation_grid(static_cast<unsigned int>(row_count));
        report_cuda_kernel_phase_launch_if_enabled(
            kernel_name,
            variant,
            "mixed_row_preparation",
            prepare,
            preparation_grid,
            dim3(evaluation_threads),
            preparation_shared
        );
        prepare<<<
            preparation_grid,
            evaluation_threads,
            preparation_shared
        >>>(
            inputs,
            plan,
            device_graph,
            first_row,
            row_count,
            workspace,
            stencil_outputs,
            mixed_stencil_outputs
        );
        check_cuda(cudaGetLastError(), "mixed path row preparation");

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
                device_graph,
                first_row,
                row_count,
                first_path,
                path_count,
                graph_configuration.path_chunk_size,
                workspace,
                launch.base_seed
            );
            check_cuda(
                cudaGetLastError(),
                "mixed path sensitivity node evaluation"
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
                workspace.graph,
                stencil_outputs,
                mixed_stencil_outputs
            );
            check_cuda(
                cudaGetLastError(),
                "mixed path sensitivity moment accumulation"
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
            workspace.graph,
            outputs,
            mixed_outputs
        );
        check_cuda(
            cudaGetLastError(),
            "mixed path sensitivity moment finalization"
        );
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
