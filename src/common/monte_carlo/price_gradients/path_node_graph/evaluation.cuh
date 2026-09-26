// Cooperative evaluation of path-dependent sensitivity graph nodes.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/path_group.cuh"
#include "common/monte_carlo/price_gradients/node_graph/row_preparation.cuh"
#include "common/monte_carlo/price_gradients/path_sensitivity_policy.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/workspace.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace path_node_graph_detail {

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

template<std::size_t NodeCapacity, typename Dynamics>
inline constexpr std::size_t scratch_bytes_per_group_v =
    has_coupled_draw_v<NodeCapacity, Dynamics>
    ? NodeCapacity * sizeof(typename Dynamics::Innovations)
    : 0U;

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
__device__ __forceinline__ void evaluate_nodes_body(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    TerminalNodeGraphWorkspace<
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>
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
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>;
    using Traits = PathScheduleTraits<Schedule>;
    using Scenario = typename Preparation::Scenario;
    using Innovations = typename Dynamics::Innovations;
    using InnovationsArray = Innovations[node_capacity];

    __shared__ Scenario scenarios[node_capacity];
    __shared__ typename Dynamics::Prepared
        dynamics[Traits::kIntervalCapacity][node_capacity];
    __shared__ pg::SensitivityStencil<4U>
        stencils[MaximumSensitivities];
    __shared__ SensitivityNodeIndices<4U>
        node_indices[MaximumSensitivities];
    __shared__ PreparedPathSchedule<Traits::kIntervalCapacity> schedule;
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
            valid_row = node_graph_detail::prepare_sensitivity_row<
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
            schedule = prepare_path_schedule<Schedule>(
                ProductPolicy::calendar(scenarios[0U].product),
                plan.time
            );
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

    for (std::size_t flat = threadIdx.x;
         flat < Traits::kIntervalCapacity * node_count;
         flat += blockDim.x) {
        const auto interval = flat / node_count;
        const auto node = flat % node_count;
        const float horizon = Traits::kExactTransition
            ? schedule.interval_years[interval]
            : plan.time.dt;
        dynamics[interval][node] =
            NodePolicy::prepare_dynamics(scenarios[node], horizon);
    }
    __syncthreads();

    const auto group = node_graph_detail::WarpPathGroup<GroupSize>::make();
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
    typename Dynamics::Prepared
        owned_dynamics[Traits::kIntervalCapacity][NodesPerWorker]{};
    typename NodePolicy::Metadata owned_metadata[NodesPerWorker]{};
    #pragma unroll
    for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
        const unsigned int node =
            group.local_lane + slot * GroupSize;
        owned_nodes[slot] = static_cast<std::uint16_t>(node);
        owns[slot] = node < node_count;
        if (!owns[slot]) continue;
        simulates[slot] = node == 0U || !scenarios[node].reuse_central;
        #pragma unroll
        for (unsigned int interval = 0U;
             interval < Traits::kIntervalCapacity;
             ++interval) {
            owned_dynamics[interval][slot] =
                dynamics[interval][node];
        }
        owned_metadata[slot] =
            NodePolicy::prepare_metadata(scenarios[node], plan.time);
        if (blockIdx.y == 0U) {
            workspace.node_metadata[
                local_row * node_capacity + node
            ] = owned_metadata[slot];
        }
    }

    for (std::size_t path = first_group_path;
         path < first_path + path_count;
         path += path_stride) {
        typename Dynamics::State states[NodesPerWorker]{};
        typename NodePolicy::Handler handlers[NodesPerWorker]{};
        bool active[NodesPerWorker]{};
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!owns[slot]) continue;
            if (simulates[slot]) {
                states[slot] =
                    Dynamics::initial(owned_dynamics[0U][slot]);
            }
            handlers[slot] =
                NodePolicy::make_handler(owned_metadata[slot]);
        }

        typename Dynamics::State central_state{};
        if (group.local_lane == 0U) central_state = states[0U];
        central_state = group.broadcast_value(central_state);
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!owns[slot]) continue;
            const auto& state = simulates[slot]
                ? states[slot]
                : central_state;
            active[slot] = handlers[slot].on_initial_value(
                NodePolicy::coordinate(owned_metadata[slot], state)
            );
        }
        group.synchronize();

        typename Dynamics::RandomContext random{};
        if (group.local_lane == 0U) random.reset(key, path);

        const auto transition = [&](unsigned int interval) {
            if constexpr (has_coupled_draw_v<node_capacity, Dynamics>) {
                auto* all_innovations =
                    reinterpret_cast<InnovationsArray*>(dynamic_shared);
                auto& group_innovations =
                    all_innovations[group.group_in_block];
                if (group.local_lane == 0U) {
                    Dynamics::template draw_coupled<node_capacity>(
                        random,
                        dynamics[interval],
                        static_cast<std::uint16_t>(node_count),
                        group_innovations
                    );
                }
                group.synchronize();
                #pragma unroll
                for (unsigned int slot = 0U;
                     slot < NodesPerWorker;
                     ++slot) {
                    if (!owns[slot] || !simulates[slot]) continue;
                    const auto node = owned_nodes[slot];
                    Dynamics::transition(
                        owned_dynamics[interval][slot],
                        group_innovations[node],
                        nullptr,
                        states[slot]
                    );
                }
            } else {
                Innovations innovations{};
                if (group.local_lane == 0U) {
                    if constexpr (requires {
                        Dynamics::draw(random, dynamics[interval][0U]);
                    }) {
                        innovations = Dynamics::draw(
                            random, dynamics[interval][0U]
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
                    if (!owns[slot] || !simulates[slot]) continue;
                    Dynamics::transition(
                        owned_dynamics[interval][slot],
                        innovations,
                        nullptr,
                        states[slot]
                    );
                }
            }
            group.synchronize();
        };

        const auto observe = [&](std::uint32_t observation) {
            if (group.local_lane == 0U) central_state = states[0U];
            central_state = group.broadcast_value(central_state);
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!owns[slot] || !active[slot]) continue;
                const auto& state = simulates[slot]
                    ? states[slot]
                    : central_state;
                active[slot] = handlers[slot].on_observation(
                    observation,
                    NodePolicy::coordinate(owned_metadata[slot], state)
                );
            }
            group.synchronize();
        };

        if constexpr (Traits::kKind == PathScheduleKind::dense) {
            for (std::uint32_t observation = 0U;
                 observation < schedule.observation_count;
                 ++observation) {
                transition(0U);
                observe(observation);
            }
        } else {
            for (std::uint32_t observation = 0U;
                 observation < schedule.observation_count;
                 ++observation) {
                const unsigned int interval =
                    Traits::kKind == PathScheduleKind::calendar
                    ? observation
                    : 0U;
                for (std::uint32_t step = 0U;
                     step < schedule.transition_counts[interval];
                     ++step) {
                    transition(interval);
                }
                observe(observation);
            }
        }

        if (group.local_lane == 0U) central_state = states[0U];
        central_state = group.broadcast_value(central_state);
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!owns[slot]) continue;
            const auto node = owned_nodes[slot];
            const auto& terminal = simulates[slot]
                ? states[slot]
                : central_state;
            workspace.node_values[
                (local_row * path_capacity + path - first_path)
                    * node_capacity
                + node
            ] = NodePolicy::finalize(
                owned_metadata[slot],
                terminal,
                handlers[slot]
            );
        }
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
__global__ void evaluate_nodes_kernel(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    TerminalNodeGraphWorkspace<
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    evaluate_nodes_body<
        Orders,
        Dynamics,
        ProductPolicy,
        Preparation,
        Schedule,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
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
    typename Schedule,
    std::size_t MaximumSensitivities,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename Tuning>
__global__ __launch_bounds__(
    Tuning::kThreadsPerBlock,
    Tuning::kMinimumBlocksPerMultiprocessor
) void bounded_evaluate_nodes_kernel(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    TerminalNodeGraphWorkspace<
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>
    > workspace,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    evaluate_nodes_body<
        Orders,
        Dynamics,
        ProductPolicy,
        Preparation,
        Schedule,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker,
        Tuning
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

}  // namespace path_node_graph_detail

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
