// Cooperative Bermudan frozen replay on graph nodes.
#pragma once

#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/launcher.cuh"
#include "common/monte_carlo/price_gradients/node_graph/path_group.cuh"
#include "product/bermudan_swaption/price_gradients/device_prepared_pricing_policy.cuh"
#include "common/monte_carlo/price_gradients/node_graph/row_preparation.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product::bermudan_swaption::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
namespace lspg =
    ::ai_factory::workbench::longstaff_schwartz::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

namespace node_graph_detail {

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
__device__ __forceinline__ void evaluate_nodes_body(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    const longstaff_schwartz::RegressionDiagnostics* diagnostics,
    mcpg::TerminalNodeGraphWorkspace<
        lspg::FrozenReplayValueNodePolicy
    > workspace,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    static_assert(mcpg::tuning::valid_node_profile_v<Tuning>);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);
    constexpr std::size_t node_capacity =
        mcpg::terminal_node_graph_node_capacity<MaximumSensitivities>();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);

    using Base = typename Policy::CentralPolicy;
    using Schedule = typename Policy::Schedule;
    using Dynamics = typename Policy::Dynamics;
    using Analytics = typename Policy::Analytics;
    using Scenario = typename Policy::Scenario;
    using Preparation = typename Policy::Preparation;
    using NodePolicy = lspg::FrozenReplayValueNodePolicy;

    __shared__ Scenario scenarios[node_capacity];
    __shared__ typename Base::PreparedRow prepared_nodes[node_capacity];
    __shared__ pg::SensitivityStencil<4U> stencils[MaximumSensitivities];
    __shared__ mcpg::SensitivityNodeIndices<4U>
        node_indices[MaximumSensitivities];
    __shared__ std::uint16_t node_count;
    __shared__ bool valid_row;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count) return;
    const auto& row = rows[local_row];

    if (threadIdx.x == 0U) {
        int error = preparation::valid;
        std::size_t error_sensitivity = 0U;
        std::uint32_t maximum_steps = 0U;
        valid_row = row.sensitivity_count > 0U
            && row.sensitivity_count <= MaximumSensitivities
            && diagnostics[local_row].fatal_failure_count == 0U;
        if (valid_row) {
            valid_row = mcpg::node_graph_detail::
                prepare_sensitivity_row_from_central<
                    Orders, Preparation, MaximumSensitivities
                >(
                    row.central,
                    row.sensitivities,
                    row.sensitivity_count,
                    row.time,
                    scenarios,
                    stencils,
                    node_indices,
                    node_count,
                    maximum_steps,
                    error,
                    error_sensitivity
                );
        } else if (diagnostics[local_row].fatal_failure_count == 0U) {
            error = preparation::unsupported_order;
        }
        if (!valid_row && diagnostics[local_row].fatal_failure_count == 0U) {
            preparation::record_error(
                row.preparation_error,
                error,
                row.result_index,
                error_sensitivity
            );
        }
        if (blockIdx.y == 0U) {
            workspace.row_status[local_row] =
                static_cast<std::uint8_t>(valid_row);
            if (valid_row) {
                for (std::size_t sensitivity = 0U;
                     sensitivity < row.sensitivity_count;
                     ++sensitivity) {
                    row.represented_stencils[
                        row.result_index * row.sensitivity_count
                            + sensitivity
                    ] = stencils[sensitivity];
                    workspace.node_indices[
                        local_row * row.sensitivity_count + sensitivity
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
        prepared_nodes[node] = Policy::prepare_node_row(
            scenarios[node],
            row.key,
            row.result_index,
            row.state_offset,
            row.paths_per_price,
            row.simulation_time
        );
        if constexpr (Policy::kTerminalForward) {
            auto* observations =
                reinterpret_cast<typename Schedule::Observation*>(
                    dynamic_shared
                ) + node * row.product.exercise_count;
            Base::prepare_observation_table(
                prepared_nodes[node], observations
            );
        }
        if (blockIdx.y == 0U) {
            workspace.node_metadata[
                local_row * node_capacity + node
            ] = typename NodePolicy::Metadata{};
        }
    }
    __syncthreads();

    const auto group =
        mcpg::node_graph_detail::WarpPathGroup<GroupSize>::make();
    constexpr unsigned int groups_per_block =
        Tuning::kThreadsPerBlock / GroupSize;
    const std::size_t first_group_path = first_path
        + static_cast<std::size_t>(blockIdx.y) * groups_per_block
        + group.group_in_block;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.y) * groups_per_block;

    std::uint16_t owned_nodes[NodesPerWorker]{};
    bool owns[NodesPerWorker]{};
    #pragma unroll
    for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
        const auto node = group.local_lane + slot * GroupSize;
        owned_nodes[slot] = static_cast<std::uint16_t>(node);
        owns[slot] = node < node_count;
    }

    for (std::size_t path = first_group_path;
         path < first_path + path_count;
         path += path_stride) {
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!owns[slot]) continue;
            const auto node = owned_nodes[slot];
            workspace.node_values[
                (local_row * path_capacity + path - first_path)
                    * node_capacity
                + node
            ] = Policy::replay_node_value(
                row, prepared_nodes[node], path
            );
        }
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Policy,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
__global__ void evaluate_nodes_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    const longstaff_schwartz::RegressionDiagnostics* diagnostics,
    mcpg::TerminalNodeGraphWorkspace<
        lspg::FrozenReplayValueNodePolicy
    > workspace,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs
) {
    evaluate_nodes_body<
        Orders,
        Policy,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
    >(
        rows,
        row_count,
        first_path,
        path_count,
        path_capacity,
        diagnostics,
        workspace,
        stencil_outputs
    );
}

}  // namespace node_graph_detail

template<
    typename BasePolicy,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
struct FrozenReplayNodeGraphPolicy : BasePolicy {
    using Base = BasePolicy;
    using CentralPolicy = typename BasePolicy::Base;
    using typename Base::PreparedRow;
    using typename Base::HostInputs;
    using BaseDeviceInputs = typename Base::DeviceInputs;
    using Workspace = mcpg::TerminalNodeGraphWorkspace<
        lspg::FrozenReplayValueNodePolicy
    >;

    struct DeviceInputs : BaseDeviceInputs {
        mcpg::TerminalNodeGraphConfiguration node_graph_configuration{};
        Workspace node_graph_workspace{};
    };

    template<typename HostPlan>
    static DeviceInputs make_device_inputs(
        const HostPlan& host,
        typename HostPlan::DeviceInputs device,
        mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
        std::size_t result_offset,
        pg::SensitivityOutputs outputs,
        mcpg::TerminalNodeGraphConfiguration graph,
        Workspace workspace
    ) {
        DeviceInputs result{};
        static_cast<BaseDeviceInputs&>(result) = Base::make_device_inputs(
            host,
            device,
            stencil_outputs,
            result_offset,
            outputs
        );
        result.node_graph_configuration = graph;
        result.node_graph_workspace = workspace;
        return result;
    }

    static std::size_t finish_batch(
        const DeviceInputs& inputs,
        const PreparedRow* rows,
        dim3 grid,
        unsigned int threads,
        std::size_t paths,
        std::size_t blocks,
        double* partials,
        const longstaff_schwartz::RegressionDiagnostics* diagnostics,
        const char* name
    ) {
        const auto evaluate = node_graph_detail::evaluate_nodes_kernel<
            Base::request_orders,
            FrozenReplayNodeGraphPolicy,
            MaximumSensitivities,
            GroupSize,
            NodesPerWorker,
            Tuning
        >;
        std::size_t shared_bytes = 0U;
        if constexpr (Base::kTerminalForward) {
            constexpr std::size_t node_capacity =
                mcpg::terminal_node_graph_node_capacity<
                    MaximumSensitivities
                >();
            shared_bytes = node_capacity
                * static_cast<std::size_t>(inputs.maximum_exercise_count)
                * sizeof(typename Base::Schedule::Observation);
        }
        return lspg::finish_frozen_replay_node_graph_batch<
            Base::request_orders,
            FrozenReplayNodeGraphPolicy,
            MaximumSensitivities
        >(
            inputs,
            rows,
            grid.y,
            threads,
            paths,
            blocks,
            partials,
            diagnostics,
            evaluate,
            Tuning::kThreadsPerBlock,
            shared_bytes,
            name
        );
    }
};

}  // namespace ai_factory::workbench::product::bermudan_swaption::price_gradients
