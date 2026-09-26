// Device preparation and cooperative evaluation of terminal graph nodes.
#pragma once

#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/node_graph/dynamics_traits.cuh"
#include "common/monte_carlo/price_gradients/node_graph/terminal_node_evaluation.cuh"
#include "common/monte_carlo/price_gradients/node_graph/row_preparation.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

namespace node_graph_detail {

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename Inputs>
__device__ __forceinline__ void evaluate_nodes_body(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    TerminalNodeGraphWorkspace<
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    static_assert(pg::requests_second_v<Orders>);
    static_assert(tuning::valid_profile_v<Tuning>);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);
    constexpr std::size_t node_capacity =
        terminal_node_graph_node_capacity<MaximumSensitivities>();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);

    using NodePolicy =
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
    using Scenario = typename Preparation::Scenario;
    using Innovations = typename Dynamics::Innovations;
    using InnovationsArray = Innovations[node_capacity];

    __shared__ Scenario scenarios[node_capacity];
    __shared__ typename Dynamics::Prepared dynamics[node_capacity];
    __shared__ pg::SensitivityStencil<4U>
        stencils[MaximumSensitivities];
    __shared__ SensitivityNodeIndices<4U>
        node_indices[MaximumSensitivities];
    __shared__ std::uint16_t node_count;
    __shared__ std::uint32_t maximum_steps;
    __shared__ bool valid_row;
    __shared__ philox::PhiloxKey key;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count) return;
    const std::size_t row = first_row + local_row;

    if (threadIdx.x == 0U) {
        int error = preparation::valid;
        std::size_t error_sensitivity = 0U;
        valid_row = plan.sensitivity_count > 0U
            && plan.sensitivity_count <= MaximumSensitivities;
        if (valid_row) {
            valid_row = prepare_sensitivity_row<
                Orders, decltype(inputs), Preparation, MaximumSensitivities
            >(
                inputs,
                plan,
                row,
                scenarios,
                stencils,
                node_indices,
                node_count,
                maximum_steps,
                error,
                error_sensitivity
            );
        } else {
            error = preparation::unsupported_order;
        }
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
                    workspace.node_indices[
                        local_row * plan.sensitivity_count + sensitivity
                    ] = node_indices[sensitivity];
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
                local_row * node_capacity + node
            ] = NodePolicy::prepare_metadata(scenarios[node], plan.time);
        }
    }
    __syncthreads();

    evaluate_terminal_node_paths<
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
        node_capacity,
        local_row,
        workspace,
        dynamic_shared
    );
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename Inputs>
__global__ void evaluate_nodes_kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    TerminalNodeGraphWorkspace<
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    evaluate_nodes_body<
        Orders,
        Dynamics,
        ProductPolicy,
        Preparation,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning,
        Inputs
    >(
        inputs,
        plan,
        first_row,
        row_count,
        first_path,
        path_count,
        path_capacity,
        workspace,
        stencil_outputs,
        base_seed
    );
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename Inputs>
__global__ __launch_bounds__(
    Tuning::kThreadsPerBlock,
    Tuning::kMinimumBlocksPerMultiprocessor
) void bounded_evaluate_nodes_kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    TerminalNodeGraphWorkspace<
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    evaluate_nodes_body<
        Orders,
        Dynamics,
        ProductPolicy,
        Preparation,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning,
        Inputs
    >(
        inputs,
        plan,
        first_row,
        row_count,
        first_path,
        path_count,
        path_capacity,
        workspace,
        stencil_outputs,
        base_seed
    );
}

}  // namespace node_graph_detail

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
