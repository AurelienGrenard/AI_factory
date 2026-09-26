// Compile-time capabilities shared by terminal sensitivity node evaluators.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

template<std::size_t NodeCapacity, typename Dynamics>
inline constexpr bool has_coupled_draw_v = requires(
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    typename Dynamics::Innovations (&innovations)[NodeCapacity]
) {
    Dynamics::template draw_coupled<NodeCapacity>(
        random, prepared, std::uint16_t{}, innovations
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

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
