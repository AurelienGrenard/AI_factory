// Cooperative propagation of one CRN interval across graph nodes.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/path_group.cuh"
#include "common/monte_carlo/price_gradients/node_graph/path_team.cuh"
#include "common/monte_carlo/price_gradients/node_graph/dynamics_traits.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

template<std::size_t NodeCapacity, typename Dynamics>
inline constexpr std::size_t coupled_interval_scratch_bytes_v = [] {
    if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
        return NodeCapacity * sizeof(typename Dynamics::TerminalAdjustment);
    } else if constexpr (has_coupled_draw_v<NodeCapacity, Dynamics>) {
        return NodeCapacity * sizeof(typename Dynamics::Innovations);
    }
    return std::size_t{0U};
}();

template<
    std::size_t NodeCapacity,
    typename Dynamics,
    unsigned int TeamSize>
inline constexpr std::size_t coupled_interval_team_scratch_bytes_v = [] {
    constexpr auto coupled = coupled_interval_scratch_bytes_v<
        NodeCapacity, Dynamics
    >;
    constexpr auto broadcast = TeamSize > 32U
        ? align_path_team_scratch_v<sizeof(typename Dynamics::Innovations)>
        : 0U;
    return coupled < broadcast ? broadcast : coupled;
}();

// Propagate an interval whose terminal time may differ by node. Continuous
// innovations retain their canonical step addresses; an independently
// aggregable component (compound-Poisson jumps for Bates) is drawn once over
// each node interval, matching the scalar pricer for the central node.
template<
    std::size_t NodeCapacity,
    typename Dynamics,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename PathGroup,
    typename StepCount>
__device__ __forceinline__ void simulate_coupled_variable_interval(
    const PathGroup& group,
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    std::uint16_t node_count,
    std::uint32_t maximum_steps,
    StepCount step_count,
    const std::uint16_t (&owned_nodes)[NodesPerWorker],
    const bool (&owns)[NodesPerWorker],
    const bool (&simulates)[NodesPerWorker],
    typename Dynamics::State (&states)[NodesPerWorker],
    unsigned char* group_scratch,
    bool active_path = true
) {
    using Innovations = typename Dynamics::Innovations;
    std::uint32_t interval_start_step = 0U;
    if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
        if (active_path && group.local_lane == 0U) {
            interval_start_step = random.step_index;
        }
    }

    for (std::uint32_t step = 0U; step < maximum_steps; ++step) {
        if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
            typename Dynamics::ContinuousInnovations innovations{};
            if (active_path && group.local_lane == 0U) {
                innovations = Dynamics::draw_continuous(
                    random, prepared[0U]
                );
            }
            innovations = group.broadcast_value(
                innovations, group_scratch
            );
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!active_path || !owns[slot] || !simulates[slot]) continue;
                const auto node = owned_nodes[slot];
                if (step < step_count(node)) {
                    Dynamics::transition_continuous(
                        prepared[node], innovations, states[slot]
                    );
                }
            }
            group.synchronize();
        } else if constexpr (has_coupled_draw_v<NodeCapacity, Dynamics>) {
            auto* innovations =
                reinterpret_cast<Innovations*>(group_scratch);
            if (active_path && group.local_lane == 0U) {
                Dynamics::template draw_coupled<NodeCapacity>(
                    random,
                    prepared,
                    node_count,
                    *reinterpret_cast<
                        Innovations (*)[NodeCapacity]
                    >(innovations)
                );
            }
            group.synchronize();
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!active_path || !owns[slot] || !simulates[slot]) continue;
                const auto node = owned_nodes[slot];
                if (step < step_count(node)) {
                    Dynamics::transition(
                        prepared[node], innovations[node], nullptr,
                        states[slot]
                    );
                }
            }
            group.synchronize();
        } else {
            Innovations innovations{};
            if (active_path && group.local_lane == 0U) {
                if constexpr (requires {
                    Dynamics::draw_equal_horizon(random);
                }) {
                    innovations = Dynamics::draw_equal_horizon(random);
                } else if constexpr (requires {
                    Dynamics::draw(random, prepared[0U]);
                }) {
                    innovations = Dynamics::draw(random, prepared[0U]);
                } else {
                    innovations = Dynamics::draw(random);
                }
            }
            innovations = group.broadcast_value(
                innovations, group_scratch
            );
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!active_path || !owns[slot] || !simulates[slot]) continue;
                const auto node = owned_nodes[slot];
                if (step < step_count(node)) {
                    Dynamics::transition(
                        prepared[node], innovations, nullptr, states[slot]
                    );
                }
            }
            group.synchronize();
        }
    }

    if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
        using Adjustment = typename Dynamics::TerminalAdjustment;
        auto* adjustments = reinterpret_cast<Adjustment*>(group_scratch);
        if (active_path && group.local_lane == 0U) {
            Dynamics::template draw_terminal_adjustments<NodeCapacity>(
                random,
                interval_start_step,
                prepared,
                node_count,
                step_count,
                *reinterpret_cast<Adjustment (*)[NodeCapacity]>(adjustments)
            );
        }
        group.synchronize();
        #pragma unroll
        for (unsigned int slot = 0U;
             slot < NodesPerWorker;
             ++slot) {
            if (!active_path || !owns[slot] || !simulates[slot]) continue;
            const auto node = owned_nodes[slot];
            Dynamics::apply_terminal_adjustment(
                prepared[node],
                adjustments[node],
                step_count(node),
                states[slot]
            );
        }
        group.synchronize();
    }
}

// The fixed-step schedule is the constant-count specialization of the same
// interval algorithm. Force-inlining exposes the constant step-count accessor
// to nvcc, so this factoring does not add a runtime branch per node.
template<
    std::size_t NodeCapacity,
    typename Dynamics,
    unsigned int GroupSize,
    unsigned int NodesPerWorker,
    typename PathGroup>
__device__ __forceinline__ void simulate_coupled_interval(
    const PathGroup& group,
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    std::uint16_t node_count,
    std::uint32_t transition_count,
    const std::uint16_t (&owned_nodes)[NodesPerWorker],
    const bool (&owns)[NodesPerWorker],
    const bool (&simulates)[NodesPerWorker],
    typename Dynamics::State (&states)[NodesPerWorker],
    unsigned char* group_scratch,
    bool active_path = true
) {
    simulate_coupled_variable_interval<
        NodeCapacity,
        Dynamics,
        GroupSize,
        NodesPerWorker
    >(
        group,
        random,
        prepared,
        node_count,
        transition_count,
        [transition_count](unsigned int) { return transition_count; },
        owned_nodes,
        owns,
        simulates,
        states,
        group_scratch,
        active_path
    );
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
