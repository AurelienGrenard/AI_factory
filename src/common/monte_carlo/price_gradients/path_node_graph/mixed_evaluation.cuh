// Multiwarp CRN evaluation of prepared mixed path sensitivity nodes.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/dynamics_traits.cuh"
#include "common/monte_carlo/price_gradients/node_graph/path_team.cuh"
#include "common/monte_carlo/price_gradients/path_node_graph/mixed_workspace.cuh"
#include "common/monte_carlo/price_gradients/path_sensitivity_policy.cuh"
#include "common/monte_carlo/price_gradients/tuning.cuh"
#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace path_node_graph_detail {

template<std::size_t NodeCapacity, typename Dynamics>
inline constexpr std::size_t mixed_path_team_scratch_bytes_v = [] {
    constexpr std::size_t bytes =
        node_graph_detail::has_coupled_draw_v<NodeCapacity, Dynamics>
        ? NodeCapacity * sizeof(typename Dynamics::Innovations)
        : sizeof(typename Dynamics::Innovations);
    return (bytes + 15U) & ~std::size_t{15U};
}();

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities,
    unsigned int TeamSize,
    unsigned int NodesPerWorker,
    typename Tuning>
__global__ void evaluate_mixed_nodes_kernel(
    pg::DeviceSensitivityGraph sensitivity_graph,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    MixedPathNodeGraphWorkspace<
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>,
        Dynamics,
        Schedule
    > workspace,
    std::uint64_t base_seed
) {
    static_assert(tuning::valid_profile_v<Tuning>);
    static_assert(Tuning::kThreadsPerBlock % TeamSize == 0U);
    constexpr auto node_capacity =
        node_graph_detail::mixed_node_graph_node_capacity<
            MaximumSensitivities, MaximumMixedSensitivities
        >();
    static_assert(TeamSize * NodesPerWorker >= node_capacity);

    using NodePolicy =
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>;
    using Traits = PathScheduleTraits<Schedule>;
    using ScheduleData = PreparedPathSchedule<Traits::kIntervalCapacity>;
    constexpr auto interval_capacity =
        path_dynamics_interval_capacity_v<Schedule>;
    constexpr unsigned int teams_per_block =
        Tuning::kThreadsPerBlock / TeamSize;
    constexpr auto scratch_stride =
        mixed_path_team_scratch_bytes_v<node_capacity, Dynamics>;

    __shared__ typename Dynamics::Prepared dynamics[node_capacity];
    __shared__ ScheduleData schedule;
    __shared__ std::uint16_t node_count;
    __shared__ philox::PhiloxKey key;
    __shared__ typename Dynamics::RandomContext
        random_contexts[teams_per_block];
    extern __shared__ __align__(16) unsigned char team_scratch_storage[];

    const std::size_t local_row = blockIdx.x;
    if (local_row >= row_count
        || workspace.graph.row_status[local_row] == 0U) {
        return;
    }
    if (threadIdx.x == 0U) {
        node_count = workspace.node_counts[local_row];
        schedule = workspace.schedules[local_row];
        key = philox::make_key(base_seed + first_row + local_row);
    }
    __syncthreads();

    const auto load_interval = [&](std::size_t interval) {
        __syncthreads();
        for (std::size_t node = threadIdx.x;
             node < node_count;
             node += blockDim.x) {
            dynamics[node] = workspace.prepared_dynamics[
                (local_row * interval_capacity + interval)
                    * sensitivity_graph.node_capacity
                + node
            ];
        }
        __syncthreads();
    };
    if constexpr (
        !Traits::kExactTransition
        || Traits::kKind != PathScheduleKind::calendar
    ) {
        load_interval(0U);
    }

    const auto team = node_graph_detail::PathTeam<TeamSize>::make();
    auto* team_scratch = team_scratch_storage
        + static_cast<std::size_t>(team.team_in_block) * scratch_stride;
    auto& random = random_contexts[team.team_in_block];
    const std::size_t team_ordinal =
        static_cast<std::size_t>(blockIdx.y) * teams_per_block
        + team.team_in_block;
    const std::size_t team_stride =
        static_cast<std::size_t>(gridDim.y) * teams_per_block;
    const std::size_t round_count =
        (path_count + team_stride - 1U) / team_stride;

    std::uint16_t owned_nodes[NodesPerWorker]{};
    bool owns[NodesPerWorker]{};
    bool simulates[NodesPerWorker]{};
    typename NodePolicy::Metadata owned_metadata[NodesPerWorker]{};
    #pragma unroll
    for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
        const unsigned int node = team.local_lane + slot * TeamSize;
        owned_nodes[slot] = static_cast<std::uint16_t>(node);
        owns[slot] = node < node_count;
        if (!owns[slot]) continue;
        simulates[slot] = workspace.simulation_flags[
            local_row * sensitivity_graph.node_capacity + node
        ] != 0U;
        owned_metadata[slot] = workspace.graph.node_metadata[
            local_row * sensitivity_graph.node_capacity + node
        ];
    }

    for (std::size_t round = 0U; round < round_count; ++round) {
        const std::size_t path = first_path + team_ordinal
            + round * team_stride;
        const bool active_path = path < first_path + path_count;
        if constexpr (
            Traits::kExactTransition
            && Traits::kKind == PathScheduleKind::calendar
        ) {
            load_interval(0U);
        }

        typename Dynamics::State states[NodesPerWorker]{};
        typename NodePolicy::Handler handlers[NodesPerWorker]{};
        bool active[NodesPerWorker]{};
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!active_path || !owns[slot]) continue;
            const auto node = owned_nodes[slot];
            if (simulates[slot]) {
                states[slot] = Dynamics::initial(dynamics[node]);
            }
            handlers[slot] = NodePolicy::make_handler(
                owned_metadata[slot]
            );
        }

        typename Dynamics::State central_state{};
        if (active_path && team.local_lane == 0U) {
            central_state = states[0U];
        }
        central_state = team.broadcast_value(central_state, team_scratch);
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!active_path || !owns[slot]) continue;
            const auto& state = simulates[slot]
                ? states[slot]
                : central_state;
            active[slot] = handlers[slot].on_initial_value(
                NodePolicy::coordinate(owned_metadata[slot], state)
            );
        }
        team.synchronize();
        if (active_path && team.local_lane == 0U) {
            random.reset(key, path);
        }
        team.synchronize();

        const auto transition = [&] {
            if constexpr (node_graph_detail::has_coupled_draw_v<
                              node_capacity, Dynamics
                          >) {
                auto* innovations = reinterpret_cast<
                    typename Dynamics::Innovations*
                >(team_scratch);
                if (active_path && team.local_lane == 0U) {
                    Dynamics::template draw_coupled<node_capacity>(
                        random,
                        dynamics,
                        node_count,
                        *reinterpret_cast<
                            typename Dynamics::Innovations (*)[node_capacity]
                        >(innovations)
                    );
                }
                team.synchronize();
                #pragma unroll
                for (unsigned int slot = 0U;
                     slot < NodesPerWorker;
                     ++slot) {
                    if (!active_path || !owns[slot] || !simulates[slot]) {
                        continue;
                    }
                    const auto node = owned_nodes[slot];
                    Dynamics::transition(
                        dynamics[node],
                        innovations[node],
                        nullptr,
                        states[slot]
                    );
                }
            } else {
                typename Dynamics::Innovations innovations{};
                if (active_path && team.local_lane == 0U) {
                    if constexpr (requires {
                        Dynamics::draw(random, dynamics[0U]);
                    }) {
                        innovations = Dynamics::draw(random, dynamics[0U]);
                    } else {
                        innovations = Dynamics::draw(random);
                    }
                }
                innovations = team.broadcast_value(
                    innovations, team_scratch
                );
                #pragma unroll
                for (unsigned int slot = 0U;
                     slot < NodesPerWorker;
                     ++slot) {
                    if (!active_path || !owns[slot] || !simulates[slot]) {
                        continue;
                    }
                    const auto node = owned_nodes[slot];
                    Dynamics::transition(
                        dynamics[node], innovations, nullptr, states[slot]
                    );
                }
            }
            team.synchronize();
        };

        const auto observe = [&](std::uint32_t observation) {
            if (active_path && team.local_lane == 0U) {
                central_state = states[0U];
            }
            central_state = team.broadcast_value(
                central_state, team_scratch
            );
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!active_path || !owns[slot] || !active[slot]) continue;
                const auto& state = simulates[slot]
                    ? states[slot]
                    : central_state;
                active[slot] = handlers[slot].on_observation(
                    observation,
                    NodePolicy::coordinate(owned_metadata[slot], state)
                );
            }
            team.synchronize();
        };

        if constexpr (Traits::kKind == PathScheduleKind::dense) {
            for (std::uint32_t observation = 0U;
                 observation < schedule.observation_count;
                 ++observation) {
                transition();
                observe(observation);
            }
        } else {
            for (std::uint32_t observation = 0U;
                 observation < schedule.observation_count;
                 ++observation) {
                if constexpr (
                    Traits::kExactTransition
                    && Traits::kKind == PathScheduleKind::calendar
                ) {
                    if (observation != 0U) load_interval(observation);
                }
                const unsigned int interval =
                    Traits::kKind == PathScheduleKind::calendar
                    ? observation
                    : 0U;
                for (std::uint32_t step = 0U;
                     step < schedule.transition_counts[interval];
                     ++step) {
                    transition();
                }
                observe(observation);
            }
        }

        if (active_path && team.local_lane == 0U) {
            central_state = states[0U];
        }
        central_state = team.broadcast_value(central_state, team_scratch);
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!active_path || !owns[slot]) continue;
            const auto node = owned_nodes[slot];
            const auto& terminal = simulates[slot]
                ? states[slot]
                : central_state;
            workspace.graph.node_values[
                (local_row * path_capacity + path - first_path)
                    * sensitivity_graph.node_capacity
                + node
            ] = NodePolicy::finalize(
                owned_metadata[slot], terminal, handlers[slot]
            );
        }
        team.synchronize();
    }
}

}  // namespace path_node_graph_detail

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
