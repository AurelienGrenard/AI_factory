// Device preparation and cooperative evaluation of mixed terminal graph nodes.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/mixed_row_preparation.cuh"
#include "common/monte_carlo/price_gradients/node_graph/terminal_node_team_evaluation.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_workspace.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil_outputs.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace node_graph_detail {

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename Inputs>
__device__ __forceinline__ void evaluate_mixed_nodes_body(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph graph,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    MixedNodeGraphWorkspace<
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs,
    std::uint64_t base_seed
) {
    static_assert(tuning::valid_profile_v<Tuning>);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);
    constexpr std::size_t node_capacity = mixed_node_graph_node_capacity<
        MaximumSensitivities, MaximumMixedSensitivities
    >();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);

    using NodePolicy =
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
    using Scenario = typename Preparation::Scenario;

    __shared__ typename Dynamics::Prepared dynamics[node_capacity];
    __shared__ pg::SensitivityStencil<4U>
        stencils[MaximumSensitivities];
    __shared__ SensitivityNodeIndices<4U>
        axis_node_indices[MaximumSensitivities];
    __shared__ pg::MixedSensitivityStencil
        mixed_stencils[MaximumMixedSensitivities];
    __shared__ MixedSensitivityNodeIndices
        mixed_node_indices[MaximumMixedSensitivities];
    __shared__ std::uint16_t node_count;
    __shared__ std::uint32_t maximum_steps;
    __shared__ bool valid_row;
    __shared__ philox::PhiloxKey key;
    constexpr unsigned int teams_per_block =
        Tuning::kThreadsPerBlock / GroupSize;
    __shared__ typename Dynamics::RandomContext
        random_contexts[teams_per_block];
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* scenarios = reinterpret_cast<Scenario*>(dynamic_shared);
    const auto scenario_bytes = (
        graph.node_capacity * sizeof(Scenario) + 15U
    ) & ~std::size_t{15U};
    auto* team_scratch = dynamic_shared + scenario_bytes;

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count) return;
    const std::size_t row = first_row + local_row;

    if (threadIdx.x == 0U) {
        int error = preparation::valid;
        std::size_t error_sensitivity = 0U;
        valid_row = prepare_mixed_sensitivity_row<
            Inputs,
            Preparation,
            MaximumSensitivities,
            MaximumMixedSensitivities
        >(
            inputs,
            plan,
            graph,
            row,
            scenarios,
            stencils,
            axis_node_indices,
            mixed_stencils,
            mixed_node_indices,
            node_count,
            maximum_steps,
            error,
            error_sensitivity
        );
        if (valid_row) {
            key = philox::make_key(base_seed + row);
        } else {
            preparation::record_error(
                stencil_outputs.error,
                error,
                row,
                error_sensitivity
            );
        }
        if (blockIdx.y == 0U) {
            workspace.row_status[local_row] =
                static_cast<std::uint8_t>(valid_row);
            if (valid_row) {
                for (std::size_t sensitivity = 0U;
                     sensitivity < plan.sensitivity_count;
                     ++sensitivity) {
                    stencil_outputs.stencils[
                        row * plan.sensitivity_count + sensitivity
                    ] = stencils[sensitivity];
                    workspace.axis_node_indices[
                        local_row * plan.sensitivity_count + sensitivity
                    ] = axis_node_indices[sensitivity];
                }
                for (std::size_t pair = 0U;
                     pair < graph.mixed_second_count;
                     ++pair) {
                    mixed_stencil_outputs.stencils[
                        row * graph.mixed_second_count + pair
                    ] = mixed_stencils[pair];
                    workspace.mixed_node_indices[
                        local_row * graph.mixed_second_count + pair
                    ] = mixed_node_indices[pair];
                }
            }
        }
    }
    __syncthreads();
    if (!valid_row) return;

    for (std::size_t node = threadIdx.x;
         node < node_count;
         node += blockDim.x) {
        dynamics[node] = NodePolicy::prepare_dynamics(
            scenarios[node], plan.time
        );
        if (blockIdx.y == 0U) {
            workspace.node_metadata[
                local_row * graph.node_capacity + node
            ] = NodePolicy::prepare_metadata(scenarios[node], plan.time);
        }
    }
    __syncthreads();

    evaluate_terminal_node_paths_by_team<
        node_capacity,
        Dynamics,
        NodePolicy,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(
        scenarios,
        dynamics,
        node_count,
        maximum_steps,
        key,
        first_path,
        path_count,
        path_capacity,
        graph.node_capacity,
        local_row,
        workspace,
        random_contexts,
        team_scratch
    );
}

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename Inputs>
__global__ void evaluate_mixed_nodes_kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph graph,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    MixedNodeGraphWorkspace<
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs,
    std::uint64_t base_seed
) {
    evaluate_mixed_nodes_body<
        Dynamics,
        ProductPolicy,
        Preparation,
        MaximumSensitivities,
        MaximumMixedSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(
        inputs,
        plan,
        graph,
        first_row,
        row_count,
        first_path,
        path_count,
        path_capacity,
        workspace,
        stencil_outputs,
        mixed_stencil_outputs,
        base_seed
    );
}

}  // namespace node_graph_detail

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
