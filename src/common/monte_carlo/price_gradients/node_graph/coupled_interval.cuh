// Cooperative propagation of one CRN interval across graph nodes.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/path_group.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/evaluation.cuh"

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
    unsigned int GroupSize,
    unsigned int NodesPerWorker>
__device__ __forceinline__ void simulate_coupled_interval(
    const WarpPathGroup<GroupSize>& group,
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    std::uint16_t node_count,
    std::uint32_t transition_count,
    const std::uint16_t (&owned_nodes)[NodesPerWorker],
    const bool (&owns)[NodesPerWorker],
    const bool (&simulates)[NodesPerWorker],
    typename Dynamics::State (&states)[NodesPerWorker],
    unsigned char* group_scratch
) {
    using Innovations = typename Dynamics::Innovations;
    std::uint32_t interval_start_step = 0U;
    if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
        if (group.local_lane == 0U) {
            interval_start_step = random.step_index;
        }
    }

    for (std::uint32_t step = 0U; step < transition_count; ++step) {
        if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
            typename Dynamics::ContinuousInnovations innovations{};
            if (group.local_lane == 0U) {
                innovations = Dynamics::draw_continuous(
                    random, prepared[0U]
                );
            }
            innovations = group.broadcast_value(innovations);
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!owns[slot] || !simulates[slot]) continue;
                Dynamics::transition_continuous(
                    prepared[owned_nodes[slot]],
                    innovations,
                    states[slot]
                );
            }
            group.synchronize();
        } else if constexpr (has_coupled_draw_v<NodeCapacity, Dynamics>) {
            auto* innovations =
                reinterpret_cast<Innovations*>(group_scratch);
            if (group.local_lane == 0U) {
                Dynamics::template draw_coupled<NodeCapacity>(
                    random, prepared, node_count, *reinterpret_cast<
                        Innovations (*)[NodeCapacity]
                    >(innovations)
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
                    prepared[node], innovations[node], nullptr, states[slot]
                );
            }
            group.synchronize();
        } else {
            Innovations innovations{};
            if (group.local_lane == 0U) {
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
            innovations = group.broadcast_value(innovations);
            #pragma unroll
            for (unsigned int slot = 0U;
                 slot < NodesPerWorker;
                 ++slot) {
                if (!owns[slot] || !simulates[slot]) continue;
                const auto node = owned_nodes[slot];
                Dynamics::transition(
                    prepared[node], innovations, nullptr, states[slot]
                );
            }
            group.synchronize();
        }
    }

    if constexpr (has_distributed_terminal_aggregation_v<Dynamics>) {
        using Adjustment = typename Dynamics::TerminalAdjustment;
        auto* adjustments = reinterpret_cast<Adjustment*>(group_scratch);
        if (group.local_lane == 0U) {
            Dynamics::template draw_terminal_adjustments<NodeCapacity>(
                random,
                interval_start_step,
                prepared,
                node_count,
                [&](unsigned int) { return transition_count; },
                *reinterpret_cast<Adjustment (*)[NodeCapacity]>(adjustments)
            );
        }
        group.synchronize();
        #pragma unroll
        for (unsigned int slot = 0U;
             slot < NodesPerWorker;
             ++slot) {
            if (!owns[slot] || !simulates[slot]) continue;
            const auto node = owned_nodes[slot];
            Dynamics::apply_terminal_adjustment(
                prepared[node],
                adjustments[node],
                transition_count,
                states[slot]
            );
        }
        group.synchronize();
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
