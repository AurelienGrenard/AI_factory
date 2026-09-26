// Host orchestration of selected mixed terminal node graphs.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/mixed_evaluation.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/mixed_reconstruction.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace mixed_node_graph_detail {

inline void validate_device_graph(
    pg::DeviceSensitivityGraph device,
    const pg::SensitivityGraphPlan& host,
    std::size_t sensitivity_count
) {
    const bool shape_mismatch =
        device.first_count != host.first.size()
        || device.diagonal_second_count != host.diagonal_second.size()
        || device.mixed_second_count != host.mixed_second.size()
        || device.coordinate_use_count != host.coordinate_uses.size()
        || device.node_capacity != host.node_capacity
        || device.coordinate_use_count != sensitivity_count;
    if (shape_mismatch
        || device.first_capacity < device.first_count
        || device.diagonal_second_capacity < device.diagonal_second_count
        || device.mixed_second_capacity < device.mixed_second_count
        || device.coordinate_use_capacity < device.coordinate_use_count) {
        throw std::invalid_argument(
            "Mixed device graph does not match its host plan."
        );
    }
    const auto validate = [](const void* pointer,
                             std::size_t count,
                             const char* label) {
        if (count != 0U) validate_device_pointer(pointer, label);
    };
    validate(device.first, device.first_count, "mixed graph first indices");
    validate(
        device.diagonal_second,
        device.diagonal_second_count,
        "mixed graph diagonal indices"
    );
    validate(
        device.mixed_second,
        device.mixed_second_count,
        "mixed graph pairs"
    );
    validate(
        device.coordinate_uses,
        device.coordinate_use_count,
        "mixed graph coordinate uses"
    );
}

}  // namespace mixed_node_graph_detail

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
    MixedTerminalNodeGraphWorkspace<
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
    mixed_node_graph_detail::validate_device_graph(
        device_graph, host_graph, plan.sensitivity_count
    );

    const auto rows = plan.result_count;
    const auto axis_stencil_count = rows * plan.sensitivity_count;
    const auto mixed_stencil_count =
        rows * host_graph.mixed_second.size();
    if (stencil_outputs.stencils == nullptr
        || stencil_outputs.capacity < axis_stencil_count
        || stencil_outputs.error == nullptr
        || mixed_stencil_outputs.stencils == nullptr
        || mixed_stencil_outputs.capacity < mixed_stencil_count) {
        throw std::invalid_argument(
            "Insufficient mixed represented-stencil output capacity."
        );
    }
    if (outputs.prices == nullptr
        || outputs.price_standard_errors == nullptr
        || outputs.price_capacity < rows
        || (host_graph.first.size() != 0U
            && (outputs.gradients == nullptr
                || outputs.gradient_standard_errors == nullptr
                || outputs.sensitivity_capacity
                    < rows * host_graph.first.size()))
        || (host_graph.diagonal_second.size() != 0U
            && (outputs.diagonal_hessians == nullptr
                || outputs.diagonal_hessian_standard_errors == nullptr
                || outputs.sensitivity_capacity
                    < rows * host_graph.diagonal_second.size()))
        || mixed_outputs.hessians == nullptr
        || mixed_outputs.standard_errors == nullptr
        || mixed_outputs.capacity < mixed_stencil_count) {
        throw std::invalid_argument(
            "Insufficient mixed terminal numerical output capacity."
        );
    }

    std::vector<pg::BufferRange> numerical_outputs{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
        pg::checked_buffer_range(
            outputs.price_standard_errors, rows, sizeof(float)
        ),
        pg::checked_buffer_range(
            mixed_outputs.hessians, mixed_stencil_count, sizeof(float)
        ),
        pg::checked_buffer_range(
            mixed_outputs.standard_errors, mixed_stencil_count, sizeof(float)
        ),
        pg::checked_buffer_range(
            mixed_stencil_outputs.stencils,
            mixed_stencil_count,
            sizeof(pg::MixedSensitivityStencil)
        ),
    };
    if (!host_graph.first.empty()) {
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.gradients, rows * host_graph.first.size(), sizeof(float)
        ));
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.gradient_standard_errors,
            rows * host_graph.first.size(),
            sizeof(float)
        ));
    }
    if (!host_graph.diagonal_second.empty()) {
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.diagonal_hessians,
            rows * host_graph.diagonal_second.size(),
            sizeof(float)
        ));
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.diagonal_hessian_standard_errors,
            rows * host_graph.diagonal_second.size(),
            sizeof(float)
        ));
    }
    validate_device_prepared_launch(
        inputs,
        plan,
        stencil_outputs,
        launch,
        numerical_outputs
    );

    const auto requirements =
        mixed_terminal_node_graph_workspace_requirements(
            plan.sensitivity_count,
            host_graph,
            launch.threads_per_block,
            graph_configuration
        );
    validate_mixed_terminal_node_graph_workspace(workspace, requirements);

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
    cudaFuncAttributes evaluation_attributes{};
    check_cuda(
        cudaFuncGetAttributes(&evaluation_attributes, evaluate),
        "mixed terminal evaluation kernel attributes"
    );
    int device = 0;
    check_cuda(cudaGetDevice(&device), "mixed terminal CUDA device");
    cudaDeviceProp device_properties{};
    check_cuda(
        cudaGetDeviceProperties(&device_properties, device),
        "mixed terminal CUDA device properties"
    );
    const auto total_shared = evaluation_shared
        + evaluation_attributes.sharedSizeBytes;
    if (total_shared > device_properties.sharedMemPerBlockOptin) {
        throw std::invalid_argument(
            "Mixed terminal sensitivity graph exceeds device shared memory."
        );
    }
    if (total_shared > device_properties.sharedMemPerBlock) {
        check_cuda(
            cudaFuncSetAttribute(
                evaluate,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(evaluation_shared)
            ),
            "mixed terminal evaluation dynamic shared memory"
        );
    }
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
