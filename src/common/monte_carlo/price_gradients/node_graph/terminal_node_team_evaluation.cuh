// Multiwarp CRN evaluation for large terminal sensitivity graphs.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/dynamics_traits.cuh"
#include "common/monte_carlo/price_gradients/node_graph/path_team.cuh"
#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

template<std::size_t NodeCapacity, typename Dynamics>
inline constexpr std::size_t path_team_scratch_bytes_v = [] {
    std::size_t bytes = distributed_node_scratch_bytes_v<
        NodeCapacity, Dynamics
    >;
    if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
        if (bytes < sizeof(typename Dynamics::ContinuousInnovations)) {
            bytes = sizeof(typename Dynamics::ContinuousInnovations);
        }
    } else if (bytes < sizeof(typename Dynamics::Innovations)) {
        bytes = sizeof(typename Dynamics::Innovations);
    }
    return (bytes + 15U) & ~std::size_t{15U};
}();

template<
    std::size_t NodeCapacity,
    typename Dynamics,
    typename NodePolicy,
    unsigned int TeamSize,
    unsigned int NodesPerWorker,
    typename Tuning,
    typename Scenario,
    typename Workspace>
__device__ __forceinline__ void evaluate_terminal_node_paths_by_team(
    const Scenario* scenarios,
    const typename Dynamics::Prepared (&dynamics)[NodeCapacity],
    std::uint16_t node_count,
    std::uint32_t maximum_steps,
    philox::PhiloxKey key,
    std::size_t first_path,
    std::size_t path_count,
    std::size_t path_capacity,
    std::size_t node_stride,
    std::size_t local_row,
    Workspace workspace,
    typename Dynamics::RandomContext* random_contexts,
    unsigned char* dynamic_shared
) {
    static_assert(Tuning::kThreadsPerBlock % TeamSize == 0U);
    static_assert(TeamSize * NodesPerWorker >= NodeCapacity);
    constexpr unsigned int teams_per_block =
        Tuning::kThreadsPerBlock / TeamSize;
    constexpr std::size_t scratch_stride =
        path_team_scratch_bytes_v<NodeCapacity, Dynamics>;

    const auto team = PathTeam<TeamSize>::make();
    auto* team_scratch = dynamic_shared
        + static_cast<std::size_t>(team.team_in_block) * scratch_stride;
    auto& random = random_contexts[team.team_in_block];
    const std::size_t first_team_path = first_path
        + static_cast<std::size_t>(blockIdx.y) * teams_per_block
        + team.team_in_block;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.y) * teams_per_block;

    std::uint16_t owned_nodes[NodesPerWorker]{};
    bool owns[NodesPerWorker]{};
    bool simulates[NodesPerWorker]{};
    std::uint32_t step_counts[NodesPerWorker]{};
    typename Dynamics::Prepared owned_dynamics[NodesPerWorker]{};
    #pragma unroll
    for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
        const unsigned int node = team.local_lane + slot * TeamSize;
        owned_nodes[slot] = static_cast<std::uint16_t>(node);
        owns[slot] = node < node_count;
        if (!owns[slot]) continue;
        simulates[slot] = node == 0U || !scenarios[node].reuse_central;
        step_counts[slot] = scenarios[node].step_count;
        owned_dynamics[slot] = dynamics[node];
    }

    for (std::size_t path = first_team_path;
         path < first_path + path_count;
         path += path_stride) {
        typename Dynamics::State states[NodesPerWorker]{};
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (owns[slot] && simulates[slot]) {
                states[slot] = Dynamics::initial(owned_dynamics[slot]);
            }
        }

        std::uint32_t interval_start_step = 0U;
        if (team.local_lane == 0U) {
            random.reset(key, path);
            if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
                interval_start_step = random.step_index;
            }
        }
        const std::uint32_t iterations =
            Dynamics::kExactTerminal ? 1U : maximum_steps;
        for (std::uint32_t step = 0U; step < iterations; ++step) {
            if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
                typename Dynamics::ContinuousInnovations innovations{};
                if (team.local_lane == 0U) {
                    innovations = Dynamics::draw_continuous(
                        random, dynamics[0U]
                    );
                }
                innovations = team.broadcast_value(
                    innovations, team_scratch
                );
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
                team.synchronize();
            } else if constexpr (has_coupled_draw_v<NodeCapacity, Dynamics>) {
                auto* innovations = reinterpret_cast<
                    typename Dynamics::Innovations*
                >(team_scratch);
                if (team.local_lane == 0U) {
                    Dynamics::template draw_coupled<NodeCapacity>(
                        random,
                        dynamics,
                        node_count,
                        *reinterpret_cast<
                            typename Dynamics::Innovations (*)[NodeCapacity]
                        >(innovations)
                    );
                }
                team.synchronize();
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
                        innovations[node],
                        Dynamics::kExactTerminal
                            ? scenarios[node].normal_weights
                            : nullptr,
                        states[slot]
                    );
                }
                team.synchronize();
            } else {
                typename Dynamics::Innovations innovations{};
                if (team.local_lane == 0U) {
                    if constexpr (requires {
                        Dynamics::draw_maturity_coupled(random);
                    }) {
                        bool needs_fourth_normal = false;
                        for (std::uint16_t node = 0U;
                             node < node_count;
                             ++node) {
                            needs_fourth_normal = needs_fourth_normal
                                || scenarios[node].normal_weights[3U] != 0.0f;
                        }
                        innovations = needs_fourth_normal
                            ? Dynamics::draw_maturity_coupled(random)
                            : Dynamics::draw(random);
                    } else if constexpr (requires {
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
                team.synchronize();
            }
        }

        if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
            auto* adjustments = reinterpret_cast<
                typename Dynamics::TerminalAdjustment*
            >(team_scratch);
            if (team.local_lane == 0U) {
                Dynamics::template draw_terminal_adjustments<NodeCapacity>(
                    random,
                    interval_start_step,
                    dynamics,
                    node_count,
                    [&](unsigned int node) {
                        return scenarios[node].step_count;
                    },
                    *reinterpret_cast<
                        typename Dynamics::TerminalAdjustment (*)[NodeCapacity]
                    >(adjustments)
                );
            }
            team.synchronize();
            #pragma unroll
            for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
                if (!owns[slot] || !simulates[slot]) continue;
                Dynamics::apply_terminal_adjustment(
                    owned_dynamics[slot],
                    adjustments[owned_nodes[slot]],
                    step_counts[slot],
                    states[slot]
                );
            }
            team.synchronize();
        }

        typename NodePolicy::NodeValue central_value{};
        if (team.local_lane == 0U) {
            central_value = NodePolicy::observe(states[0U]);
        }
        central_value = team.broadcast_value(
            central_value, team_scratch
        );
        #pragma unroll
        for (unsigned int slot = 0U; slot < NodesPerWorker; ++slot) {
            if (!owns[slot]) continue;
            const auto node = owned_nodes[slot];
            const auto value = simulates[slot]
                ? NodePolicy::observe(states[slot])
                : central_value;
            const auto local_path = path - first_path;
            workspace.node_values[
                (local_row * path_capacity + local_path) * node_stride + node
            ] = value;
        }
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
