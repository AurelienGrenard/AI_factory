// Cooperative frozen replay of selected mixed American graph nodes.
#pragma once

#include "common/equity/discount.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/mixed_launcher.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/mixed_preparation.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_replay_node_graph/value_policy.cuh"
#include "common/monte_carlo/price_gradients/node_graph/coupled_interval.cuh"
#include "common/monte_carlo/price_gradients/node_graph/kernel_shared_memory.cuh"
#include "common/monte_carlo/price_gradients/node_graph/path_team.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/american_option/price_gradients/frozen_replay_mixed_workspace.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product::american_option::price_gradients {

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
    constexpr std::size_t node_capacity =
        mcpg::node_graph_detail::mixed_node_graph_node_capacity<
            MaximumSensitivities, MaximumMixedSensitivities
        >();

    using CentralPolicy = typename Policy::CentralPolicy;
    using Replay = typename Policy::Replay;
    using Dynamics = typename Replay::Dynamics;
    using Scenario = typename Policy::Scenario;
    using Preparation = typename Policy::Preparation;
    using NodePolicy = lspg::FrozenReplayValueNodePolicy;
    constexpr auto interval_capacity =
        mixed_workspace_detail::interval_capacity_v<Replay>;

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
        const bool report_preparation_error =
            diagnostics[local_row].fatal_failure_count == 0U;
        const bool preparation_allowed = report_preparation_error
            && Policy::initial_exercise_decision(row)
                != longstaff_schwartz::InitialExerciseDecision::invalid;
        valid_row = lspg::prepare_frozen_replay_mixed_row<
            Preparation,
            MaximumSensitivities,
            MaximumMixedSensitivities
        >(
            row,
            preparation_allowed,
            report_preparation_error,
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

    const float central_rate = scenarios[0U].model.risk_free_rate;
    const lspg::FrozenExerciseReplayTime replay_time{
        row.time.dt,
        row.first_exercise_time,
        row.exercise_interval,
    };
    const auto node_stride = sensitivity_graph.node_capacity;
    for (std::size_t node = threadIdx.x;
         node < node_count;
         node += blockDim.x) {
        const auto row_node = local_row * node_stride + node;
        if constexpr (Policy::kFrozenRegressionPolicy) {
            workspace.prepared_nodes[row_node] = Policy::prepare_node_row(
                scenarios[node],
                row.key,
                row.result_index,
                row.state_offset,
                row.paths_per_price,
                row.simulation_time
            );
        } else {
            const auto prepared = Replay::prepare_graph_node(
                scenarios[node], replay_time
            );
            if constexpr (Replay::kExactTransitionReplay) {
                workspace.prepared_dynamics[
                    local_row * interval_capacity * node_capacity + node
                ] = prepared.initial;
                workspace.prepared_dynamics[
                    (local_row * interval_capacity + 1U) * node_capacity + node
                ] = prepared.regular;
            } else {
                workspace.prepared_dynamics[
                    local_row * interval_capacity * node_capacity + node
                ] = prepared;
            }
            workspace.spot_scales[row_node] = scenarios[node].spot_scale;
            workspace.strikes[row_node] = scenarios[node].product.strike;
            workspace.initial_spots[row_node] = scenarios[node].model.spot;
            workspace.reuse_central[row_node] = static_cast<std::uint8_t>(
                node != 0U && scenarios[node].reuse_central
            );
            if (scenarios[node].model.risk_free_rate == central_rate) {
                workspace.initial_discounts[row_node] = row.initial_discount;
                workspace.exercise_discounts[row_node] = row.exercise_discount;
            } else {
                workspace.initial_discounts[row_node] =
                    equity::constant_rate_discount_factor(
                        scenarios[node].model, row.first_exercise_time
                    );
                workspace.exercise_discounts[row_node] =
                    equity::constant_rate_discount_factor(
                        scenarios[node].model, row.exercise_interval
                    );
            }
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

    using Replay = typename Policy::Replay;
    using Dynamics = typename Replay::Dynamics;
    constexpr auto interval_capacity =
        mixed_workspace_detail::interval_capacity_v<Replay>;

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count
        || workspace.graph.row_status[local_row] == 0U) {
        return;
    }
    const auto& row = rows[local_row];
    const auto node_count = workspace.node_counts[local_row];
    const auto node_stride = sensitivity_graph.node_capacity;
    const auto row_node_offset = local_row * node_stride;

    if constexpr (Policy::kFrozenRegressionPolicy) {
        const auto group =
            mcpg::node_graph_detail::PathTeam<GroupSize>::make();
        constexpr unsigned int groups_per_block =
            Tuning::kThreadsPerBlock / GroupSize;
        const std::size_t first_group_path = first_path
            + static_cast<std::size_t>(blockIdx.y) * groups_per_block
            + group.team_in_block;
        const std::size_t path_stride =
            static_cast<std::size_t>(gridDim.y) * groups_per_block;
        const auto* prepared_nodes = workspace.prepared_nodes
            + row_node_offset;

        for (std::size_t path = first_group_path;
             path < first_path + path_count;
             path += path_stride) {
            #pragma unroll
            for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
                const auto node = group.local_lane + slot * GroupSize;
                if (node >= node_count) continue;
                workspace.graph.node_values[
                    (local_row * path_capacity + path - first_path)
                        * node_stride + node
                ] = Policy::replay_node_value(
                    row, prepared_nodes[node], path
                );
            }
        }
        return;
    } else {
        const auto* initial_dynamics_pointer = workspace.prepared_dynamics
            + local_row * interval_capacity * node_capacity;
        const auto& initial_dynamics = *reinterpret_cast<
            const typename Dynamics::Prepared (*)[node_capacity]
        >(initial_dynamics_pointer);
        const auto* regular_dynamics_pointer = Replay::kExactTransitionReplay
            ? initial_dynamics_pointer + node_capacity
            : initial_dynamics_pointer;
        const auto& regular_dynamics = *reinterpret_cast<
            const typename Dynamics::Prepared (*)[node_capacity]
        >(regular_dynamics_pointer);

        const auto group =
            mcpg::node_graph_detail::PathTeam<GroupSize>::make();
        constexpr unsigned int groups_per_block =
            Tuning::kThreadsPerBlock / GroupSize;
        constexpr std::size_t scratch_bytes_per_group =
            mcpg::node_graph_detail::coupled_interval_team_scratch_bytes_v<
                node_capacity, Dynamics, GroupSize
            >;
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        unsigned char* const group_scratch = dynamic_shared
            + group.team_in_block * scratch_bytes_per_group;
        const std::size_t first_group_path = first_path
            + static_cast<std::size_t>(blockIdx.y) * groups_per_block
            + group.team_in_block;
        const std::size_t path_stride =
            static_cast<std::size_t>(gridDim.y) * groups_per_block;

        std::uint16_t owned_nodes[NodesPerWorker]{};
        bool owns[NodesPerWorker]{};
        bool simulates[NodesPerWorker]{};
        float spot_scales[NodesPerWorker]{};
        float strikes[NodesPerWorker]{};
        float initial_spots[NodesPerWorker]{};
        float initial_discounts[NodesPerWorker]{};
        float exercise_discounts[NodesPerWorker]{};
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            const auto node = group.local_lane + slot * GroupSize;
            owned_nodes[slot] = static_cast<std::uint16_t>(node);
            owns[slot] = node < node_count;
            if (!owns[slot]) continue;
            const auto row_node = row_node_offset + node;
            simulates[slot] = node != 0U
                && workspace.reuse_central[row_node] == 0U;
            spot_scales[slot] = workspace.spot_scales[row_node];
            strikes[slot] = workspace.strikes[row_node];
            initial_spots[slot] = workspace.initial_spots[row_node];
            initial_discounts[slot] = workspace.initial_discounts[row_node];
            exercise_discounts[slot] = workspace.exercise_discounts[row_node];
        }

        const bool initial_exercise =
            Policy::initial_exercise_decision(row)
                == longstaff_schwartz::InitialExerciseDecision::exercise;
        for (std::size_t path = first_group_path;
             path < first_path + path_count;
             path += path_stride) {
            const auto exercise = row.exercises[path];
            typename Dynamics::State states[NodesPerWorker]{};
            #pragma unroll
            for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
                if (simulates[slot]) {
                    states[slot] = Dynamics::initial(
                        initial_dynamics[owned_nodes[slot]]
                    );
                }
            }

            if (!initial_exercise) {
                typename Dynamics::RandomContext random{};
                if (group.local_lane == 0U) random.reset(row.key, path);
                mcpg::node_graph_detail::simulate_coupled_interval<
                    node_capacity, Dynamics, GroupSize, NodesPerWorker
                >(
                    group,
                    random,
                    initial_dynamics,
                    node_count,
                    Replay::kExactTransitionReplay
                        ? 1U
                        : row.initial_transition_count,
                    owned_nodes,
                    owns,
                    simulates,
                    states,
                    group_scratch
                );
                for (std::uint32_t observation = 0U;
                     observation < exercise.observation;
                     ++observation) {
                    mcpg::node_graph_detail::simulate_coupled_interval<
                        node_capacity, Dynamics, GroupSize, NodesPerWorker
                    >(
                        group,
                        random,
                        regular_dynamics,
                        node_count,
                        Replay::kExactTransitionReplay
                            ? 1U
                            : row.transitions_per_exercise,
                        owned_nodes,
                        owns,
                        simulates,
                        states,
                        group_scratch
                    );
                }
            }

            #pragma unroll
            for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
                if (!owns[slot]) continue;
                const auto node = owned_nodes[slot];
                float spot = initial_spots[slot];
                if (!initial_exercise) {
                    spot = simulates[slot]
                        ? Dynamics::spot(states[slot]) * spot_scales[slot]
                        : exercise.spot * spot_scales[slot];
                }
                float value = payoff::vanilla_option_payoff<Policy::kSide>(
                    spot, strikes[slot]
                );
                if (!initial_exercise) {
                    for (std::uint32_t date = 0U;
                         date < exercise.observation;
                         ++date) {
                        value = exercise_discounts[slot] * value;
                    }
                    value = initial_discounts[slot] * value;
                }
                workspace.graph.node_values[
                    (local_row * path_capacity + path - first_path)
                        * node_stride
                    + node
                ] = value;
            }
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
    using Replay = typename Base::Replay;
    using Scenario = typename Base::Scenario;
    using Preparation = typename Base::Preparation;
    using BaseDeviceInputs = typename Base::DeviceInputs;
    using Workspace = std::conditional_t<
        Base::kFrozenRegressionPolicy,
        FrozenRegressionMixedWorkspace<CentralPolicy>,
        FrozenExerciseMixedWorkspace<Replay>
    >;
    using WorkspaceLayout = std::conditional_t<
        Base::kFrozenRegressionPolicy,
        FrozenRegressionMixedWorkspaceLayout,
        FrozenExerciseMixedWorkspaceLayout
    >;
    using WorkspaceRequirements = std::conditional_t<
        Base::kFrozenRegressionPolicy,
        FrozenRegressionMixedWorkspaceRequirements,
        FrozenExerciseMixedWorkspaceRequirements
    >;

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
        if constexpr (Base::kFrozenRegressionPolicy) {
            return frozen_regression_mixed_workspace_layout<CentralPolicy>(
                host.sensitivity_count(), host.sensitivity_graph,
                reduction_threads, configuration
            );
        } else {
            return frozen_exercise_mixed_workspace_layout<Replay>(
                host.sensitivity_count(),
                host.sensitivity_graph,
                reduction_threads,
                configuration,
                mcpg::node_graph_detail::mixed_node_graph_node_capacity<
                    MaximumSensitivities, MaximumMixedSensitivities
                >()
            );
        }
    }

    static Workspace make_mixed_workspace(
        void* storage,
        std::size_t storage_bytes,
        const WorkspaceLayout& layout
    ) {
        if constexpr (Base::kFrozenRegressionPolicy) {
            return make_frozen_regression_mixed_workspace<CentralPolicy>(
                storage, storage_bytes, layout
            );
        } else {
            return make_frozen_exercise_mixed_workspace<Replay>(
                storage, storage_bytes, layout
            );
        }
    }

    static void validate_mixed_workspace(
        Workspace workspace,
        const WorkspaceRequirements& required
    ) {
        if constexpr (Base::kFrozenRegressionPolicy) {
            validate_frozen_regression_mixed_workspace(workspace, required);
        } else {
            validate_frozen_exercise_mixed_workspace(workspace, required);
        }
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
        constexpr std::size_t node_capacity =
            mcpg::node_graph_detail::mixed_node_graph_node_capacity<
                MaximumSensitivities, MaximumMixedSensitivities
            >();
        constexpr unsigned int groups_per_block =
            Tuning::kThreadsPerBlock / GroupSize;
        constexpr std::size_t scratch_per_group =
            mcpg::node_graph_detail::coupled_interval_team_scratch_bytes_v<
                node_capacity, typename Replay::Dynamics, GroupSize
            >;
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
        constexpr std::size_t evaluation_shared_bytes =
            Base::kFrozenRegressionPolicy
                ? 0U
                : groups_per_block * scratch_per_group;
        mcpg::configure_node_graph_dynamic_shared_memory(
            prepare,
            preparation_shared_bytes,
            "American mixed preparation kernel attributes",
            "American mixed preparation CUDA device",
            "American mixed preparation CUDA device properties",
            "American mixed preparation exceeds device shared memory.",
            "American mixed preparation dynamic shared memory"
        );
        mcpg::configure_node_graph_dynamic_shared_memory(
            evaluate,
            evaluation_shared_bytes,
            "American mixed frozen-replay kernel attributes",
            "American mixed frozen-replay CUDA device",
            "American mixed frozen-replay CUDA device properties",
            "American mixed frozen-replay exceeds device shared memory.",
            "American mixed frozen-replay dynamic shared memory"
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
            evaluation_shared_bytes,
            name
        );
    }
};

}  // namespace ai_factory::workbench::product::american_option::price_gradients
