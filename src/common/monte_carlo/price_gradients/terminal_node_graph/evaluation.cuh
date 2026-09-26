// Device preparation and cooperative evaluation of terminal graph nodes.
#pragma once

#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/node_graph/path_group.cuh"
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

template<std::size_t NodeCapacity, typename Dynamics>
inline constexpr bool has_coupled_draw_v = requires(
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    typename Dynamics::Innovations (&innovations)[NodeCapacity]
) {
    Dynamics::template draw_coupled<NodeCapacity>(
        random, prepared, std::uint8_t{}, innovations
    );
};

template<typename Dynamics>
inline constexpr bool has_distributed_terminal_aggregation_v = [] {
    if constexpr (requires {
        Dynamics::kHasDistributedTerminalAggregation;
    }) {
        return Dynamics::kHasDistributedTerminalAggregation;
    }
    return false;
}();

template<std::size_t NodeCapacity, typename Dynamics>
inline constexpr std::size_t distributed_node_scratch_bytes_v = [] {
    if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
        return NodeCapacity * sizeof(typename Dynamics::TerminalAdjustment);
    } else if constexpr (has_coupled_draw_v<NodeCapacity, Dynamics>) {
        return NodeCapacity * sizeof(typename Dynamics::Innovations);
    }
    return std::size_t{0U};
}();

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

    const auto group = WarpPathGroup<GroupSize>::make();
    constexpr unsigned int groups_per_block =
        Tuning::kThreadsPerBlock / GroupSize;
    const std::size_t first_group_path =
        first_path
        + static_cast<std::size_t>(blockIdx.y) * groups_per_block
        + group.group_in_block;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.y) * groups_per_block;

    std::uint16_t owned_nodes[NodesPerWorker]{};
    bool owns[NodesPerWorker]{};
    bool simulates[NodesPerWorker]{};
    std::uint32_t step_counts[NodesPerWorker]{};
    typename Dynamics::Prepared owned_dynamics[NodesPerWorker]{};
    #pragma unroll
    for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
        const unsigned int node =
            group.local_lane + slot * GroupSize;
        owned_nodes[slot] = static_cast<std::uint16_t>(node);
        owns[slot] = node < node_count;
        if (!owns[slot]) continue;
        simulates[slot] =
            node == 0U || !scenarios[node].reuse_central;
        step_counts[slot] = scenarios[node].step_count;
        owned_dynamics[slot] = dynamics[node];
    }

    for (std::size_t path = first_group_path;
         path < first_path + path_count;
         path += path_stride) {
        typename Dynamics::State states[NodesPerWorker]{};
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (owns[slot] && simulates[slot]) {
                states[slot] =
                    Dynamics::initial(owned_dynamics[slot]);
            }
        }

        typename Dynamics::RandomContext random{};
        std::uint32_t interval_start_step = 0U;
        if (group.local_lane == 0U) {
            random.reset(key, path);
            if constexpr (
                has_distributed_terminal_aggregation_v<Dynamics>
            ) {
                interval_start_step = random.step_index;
            }
        }
        const std::uint32_t iterations =
            Dynamics::kExactTerminal ? 1U : maximum_steps;
        for (std::uint32_t step = 0U; step < iterations; ++step) {
            if constexpr (
                has_distributed_terminal_aggregation_v<Dynamics>
            ) {
                typename Dynamics::ContinuousInnovations innovations{};
                if (group.local_lane == 0U) {
                    innovations = Dynamics::draw_continuous(
                        random, dynamics[0U]
                    );
                }
                innovations = group.broadcast_value(innovations);
                #pragma unroll
                for (unsigned int slot = 0U;
                     slot < NodesPerWorker;
                     ++slot) {
                    if (!owns[slot] || !simulates[slot]
                        || step >= step_counts[slot]) {
                        continue;
                    }
                    Dynamics::transition_continuous(
                        owned_dynamics[slot], innovations, states[slot]
                    );
                }
                group.synchronize();
            } else if constexpr (
                has_coupled_draw_v<node_capacity, Dynamics>
            ) {
                auto* all_innovations =
                    reinterpret_cast<InnovationsArray*>(dynamic_shared);
                auto& group_innovations =
                    all_innovations[group.group_in_block];
                if (group.local_lane == 0U) {
                    Dynamics::template draw_coupled<node_capacity>(
                        random,
                        dynamics,
                        static_cast<std::uint8_t>(node_count),
                        group_innovations
                    );
                }
                group.synchronize();
                #pragma unroll
                for (unsigned int slot = 0U;
                     slot < NodesPerWorker;
                     ++slot) {
                    if (!owns[slot] || !simulates[slot]
                        || (!Dynamics::kExactTerminal
                            && step >= step_counts[slot])) {
                        continue;
                    }
                    const auto node = owned_nodes[slot];
                    const auto innovations = group_innovations[node];
                    Dynamics::transition(
                        owned_dynamics[slot],
                        innovations,
                        Dynamics::kExactTerminal
                            ? scenarios[node].normal_weights
                            : nullptr,
                        states[slot]
                    );
                }
                group.synchronize();
            } else {
                Innovations innovations{};
                if (group.local_lane == 0U) {
                    if constexpr (requires {
                        Dynamics::draw(random, dynamics[0U]);
                    }) {
                        innovations = Dynamics::draw(
                            random, dynamics[0U]
                        );
                    } else {
                        innovations = Dynamics::draw(random);
                    }
                }
                innovations = group.broadcast_value(innovations);
                #pragma unroll
                for (unsigned int slot = 0U;
                     slot < NodesPerWorker;
                     ++slot) {
                    if (!owns[slot] || !simulates[slot]
                        || (!Dynamics::kExactTerminal
                            && step >= step_counts[slot])) {
                        continue;
                    }
                    const auto node = owned_nodes[slot];
                    Dynamics::transition(
                        owned_dynamics[slot],
                        innovations,
                        Dynamics::kExactTerminal
                            ? scenarios[node].normal_weights
                            : nullptr,
                        states[slot]
                    );
                }
                group.synchronize();
            }
        }
        if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
            using Adjustment = typename Dynamics::TerminalAdjustment;
            using AdjustmentArray = Adjustment[node_capacity];
            auto* all_adjustments =
                reinterpret_cast<AdjustmentArray*>(dynamic_shared);
            auto& group_adjustments =
                all_adjustments[group.group_in_block];
            if (group.local_lane == 0U) {
                Dynamics::template draw_terminal_adjustments<node_capacity>(
                    random,
                    interval_start_step,
                    dynamics,
                    static_cast<std::uint8_t>(node_count),
                    [&](unsigned int node) {
                        return scenarios[node].step_count;
                    },
                    group_adjustments
                );
            }
            group.synchronize();
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!owns[slot] || !simulates[slot]) continue;
                Dynamics::apply_terminal_adjustment(
                    owned_dynamics[slot],
                    group_adjustments[owned_nodes[slot]],
                    step_counts[slot],
                    states[slot]
                );
            }
            group.synchronize();
        }

        typename NodePolicy::NodeValue central_value{};
        if (group.local_lane == 0U) {
            central_value = NodePolicy::observe(states[0U]);
        }
        central_value = group.broadcast_value(central_value);
        #pragma unroll
        for (unsigned int slot = 0U;
             slot < NodesPerWorker;
             ++slot) {
            if (!owns[slot]) continue;
            const auto node = owned_nodes[slot];
            const auto value = simulates[slot]
                ? NodePolicy::observe(states[slot])
                : central_value;
            const auto local_path = path - first_path;
            workspace.node_values[
                (local_row * path_capacity + local_path)
                    * node_capacity
                + node
            ] = value;
        }
    }
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
