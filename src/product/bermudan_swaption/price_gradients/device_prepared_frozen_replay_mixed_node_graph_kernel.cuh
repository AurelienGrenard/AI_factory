// Cooperative Bermudan frozen replay on graph nodes.
#pragma once

#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/mixed_launcher.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/mixed_preparation.cuh"
#include "common/monte_carlo/price_gradients/node_graph/kernel_shared_memory.cuh"
#include "common/monte_carlo/price_gradients/node_graph/path_team.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"
#include "product/bermudan_swaption/price_gradients/device_prepared_pricing_policy.cuh"
#include "product/bermudan_swaption/price_gradients/frozen_replay_mixed_workspace.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product::bermudan_swaption::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
namespace lspg =
    ::ai_factory::workbench::longstaff_schwartz::price_gradients;
namespace node_graph_detail {

template<
    typename Policy,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    typename Tuning>
__global__ void prepare_mixed_nodes_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    const longstaff_schwartz::RegressionDiagnostics* diagnostics,
    pg::DeviceSensitivityGraph sensitivity_graph,
    typename Policy::Workspace workspace,
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs
) {
    static_assert(mcpg::tuning::valid_node_profile_v<Tuning>);

    using Base = typename Policy::CentralPolicy;
    using Schedule = typename Policy::Schedule;
    using Scenario = typename Policy::Scenario;
    using Preparation = typename Policy::Preparation;
    using NodePolicy = lspg::FrozenReplayValueNodePolicy;

    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* scenarios = reinterpret_cast<Scenario*>(dynamic_shared);
    __shared__ pg::SensitivityStencil<4U> stencils[MaximumSensitivities];
    __shared__ mcpg::SensitivityNodeIndices<4U>
        axis_node_indices[MaximumSensitivities];
    __shared__ pg::MixedSensitivityStencil
        mixed_stencils[MaximumMixedSensitivities];
    __shared__ mcpg::MixedSensitivityNodeIndices
        mixed_node_indices[MaximumMixedSensitivities];
    __shared__ std::uint16_t node_count;
    __shared__ bool valid_row;

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count) return;
    const auto& row = rows[local_row];

    if (threadIdx.x == 0U) {
        const bool preparation_allowed =
            diagnostics[local_row].fatal_failure_count == 0U;
        valid_row = lspg::prepare_frozen_replay_mixed_row<
            Preparation,
            MaximumSensitivities,
            MaximumMixedSensitivities
        >(
            row,
            preparation_allowed,
            preparation_allowed,
            sensitivity_graph,
            workspace.graph,
            mixed_stencil_outputs,
            scenarios,
            stencils,
            axis_node_indices,
            mixed_stencils,
            mixed_node_indices,
            local_row,
            node_count
        );
        if (valid_row) {
            workspace.node_counts[local_row] = node_count;
        }
    }
    __syncthreads();
    if (!valid_row) return;

    const auto node_stride = sensitivity_graph.node_capacity;
    for (std::size_t node = threadIdx.x;
         node < node_count;
         node += blockDim.x) {
        const auto row_node = local_row * node_stride + node;
        auto& prepared = workspace.prepared_nodes[row_node];
        prepared = Policy::prepare_node_row(
            scenarios[node],
            row.key,
            row.result_index,
            row.state_offset,
            row.paths_per_price,
            row.simulation_time
        );
        if constexpr (Policy::kTerminalForward) {
            auto* observations = workspace.observations
                + row_node * workspace.observation_stride;
            Base::prepare_observation_table(prepared, observations);
        }
        workspace.graph.node_metadata[row_node] =
            typename NodePolicy::Metadata{};
    }
}

template<
    typename Policy,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
__global__ void evaluate_mixed_nodes_kernel(
    const typename Policy::PreparedRow* rows,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    const longstaff_schwartz::RegressionDiagnostics*,
    pg::DeviceSensitivityGraph sensitivity_graph,
    typename Policy::Workspace workspace,
    mcpg::DevicePreparedStencilOutputs<4U>,
    pg::MixedSensitivityStencilOutputs
) {
    static_assert(mcpg::tuning::valid_node_profile_v<Tuning>);
    static_assert(Tuning::kThreadsPerBlock % GroupSize == 0U);
    constexpr std::size_t node_capacity =
        mcpg::node_graph_detail::mixed_node_graph_node_capacity<
            MaximumSensitivities, MaximumMixedSensitivities
        >();
    static_assert(GroupSize * NodesPerWorker >= node_capacity);

    using Base = typename Policy::CentralPolicy;
    using Schedule = typename Policy::Schedule;
    using Dynamics = typename Policy::Dynamics;
    using Analytics = typename Policy::Analytics;

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count
        || workspace.graph.row_status[local_row] == 0U) {
        return;
    }
    const auto& row = rows[local_row];
    const auto node_count = workspace.node_counts[local_row];
    const auto node_stride = sensitivity_graph.node_capacity;
    const auto* prepared_nodes = workspace.prepared_nodes
        + local_row * node_stride;

    const auto group =
        mcpg::node_graph_detail::PathTeam<GroupSize>::make();
    constexpr unsigned int groups_per_block =
        Tuning::kThreadsPerBlock / GroupSize;
    const std::size_t first_group_path = first_path
        + static_cast<std::size_t>(blockIdx.y) * groups_per_block
        + group.team_in_block;
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
            workspace.graph.node_values[
                (local_row * path_capacity + path - first_path)
                    * node_stride
                + node
            ] = Policy::replay_node_value(
                row, prepared_nodes[node], path
            );
        }
    }
}

}  // namespace node_graph_detail

template<
    typename BasePolicy,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
struct FrozenReplayMixedNodeGraphPolicy : BasePolicy {
    using Base = BasePolicy;
    using CentralPolicy = typename BasePolicy::Base;
    using typename Base::PreparedRow;
    using typename Base::HostInputs;
    using Scenario = typename Base::Scenario;
    using Preparation = typename Base::Preparation;
    using Schedule = typename Base::Schedule;
    using Dynamics = typename Base::Dynamics;
    using Analytics = typename Base::Analytics;
    using BaseDeviceInputs = typename Base::DeviceInputs;
    using Workspace = FrozenReplayMixedWorkspace<
        CentralPolicy, Base::kTerminalForward
    >;
    using WorkspaceLayout = FrozenReplayMixedWorkspaceLayout;

    struct DeviceInputs : BaseDeviceInputs {
        mcpg::TerminalNodeGraphConfiguration node_graph_configuration{};
        pg::DeviceSensitivityGraph sensitivity_graph{};
        pg::MixedSensitivityStencilOutputs mixed_stencil_outputs{};
        pg::MixedSensitivityOutputs mixed_outputs{};
        std::size_t task_count = 0U;
        Workspace node_graph_workspace{};

        void validate(std::size_t results) const {
            BaseDeviceInputs::validate_inputs_and_tasks(results);
            mcpg::validate_mixed_output_views(
                this->total_result_count,
                sensitivity_graph.first_count,
                sensitivity_graph.diagonal_second_count,
                sensitivity_graph.mixed_second_count,
                this->outputs,
                mixed_outputs,
                mixed_stencil_outputs
            );
        }
    };

    template<typename HostPlan>
    static WorkspaceLayout make_mixed_workspace_layout(
        const HostPlan& host,
        unsigned int reduction_threads,
        mcpg::TerminalNodeGraphConfiguration configuration
    ) {
        return frozen_replay_mixed_workspace_layout<
            CentralPolicy, Base::kTerminalForward
        >(
            host.sensitivity_count(),
            host.sensitivity_graph,
            reduction_threads,
            configuration,
            host.maximum_exercise_count
        );
    }

    static Workspace make_mixed_workspace(
        void* storage,
        std::size_t storage_bytes,
        const WorkspaceLayout& layout
    ) {
        return make_frozen_replay_mixed_workspace<
            CentralPolicy, Base::kTerminalForward
        >(
            storage, storage_bytes, layout
        );
    }

    static void validate_mixed_workspace(
        Workspace workspace,
        const FrozenReplayMixedWorkspaceRequirements& required
    ) {
        validate_frozen_replay_mixed_workspace(workspace, required);
    }

    template<typename HostPlan>
    static DeviceInputs make_device_inputs(
        const HostPlan& host,
        typename HostPlan::DeviceInputs device,
        mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs,
        pg::MixedSensitivityStencilOutputs mixed_stencil_outputs,
        std::size_t result_offset,
        pg::SensitivityOutputs outputs,
        pg::MixedSensitivityOutputs mixed_outputs,
        mcpg::TerminalNodeGraphConfiguration graph,
        pg::DeviceSensitivityGraph sensitivity_graph,
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
        result.sensitivity_graph = sensitivity_graph;
        result.mixed_stencil_outputs = mixed_stencil_outputs;
        result.mixed_outputs = mixed_outputs;
        result.task_count = sensitivity_graph.output_count() - 1U;
        result.node_graph_workspace = workspace;
        return result;
    }

    static std::size_t moment_value_count(const HostInputs&) {
        return 2U;
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
        (void)blocks;
        (void)partials;
        const auto prepare = node_graph_detail::prepare_mixed_nodes_kernel<
            FrozenReplayMixedNodeGraphPolicy,
            MaximumSensitivities,
            MaximumMixedSensitivities,
            Tuning
        >;
        const auto evaluate = node_graph_detail::evaluate_mixed_nodes_kernel<
            FrozenReplayMixedNodeGraphPolicy,
            MaximumSensitivities,
            MaximumMixedSensitivities,
            GroupSize,
            NodesPerWorker,
            Tuning
        >;
        const std::size_t preparation_shared_bytes =
            (inputs.sensitivity_graph.node_capacity * sizeof(Scenario) + 15U)
            & ~std::size_t{15U};
        mcpg::configure_node_graph_dynamic_shared_memory(
            prepare,
            preparation_shared_bytes,
            "Bermudan mixed preparation kernel attributes",
            "Bermudan mixed preparation CUDA device",
            "Bermudan mixed preparation CUDA device properties",
            "Bermudan mixed preparation exceeds device shared memory.",
            "Bermudan mixed preparation dynamic shared memory"
        );
        return lspg::finish_frozen_replay_mixed_node_graph_batch<
            FrozenReplayMixedNodeGraphPolicy
        >(
            inputs,
            rows,
            grid.y,
            threads,
            paths,
            diagnostics,
            prepare,
            Tuning::kThreadsPerBlock,
            preparation_shared_bytes,
            evaluate,
            Tuning::kThreadsPerBlock,
            0U,
            name
        );
    }
};

}  // namespace ai_factory::workbench::product::bermudan_swaption::price_gradients
