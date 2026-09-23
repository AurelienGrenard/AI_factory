// Compile-time selection between one shared innovation and node-wise coupling.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

template<std::size_t NodeCapacity, typename Dynamics, typename Apply>
__device__ __forceinline__ void draw_and_apply_node_innovations(
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    Apply&& apply
) {
    if constexpr (requires(
        typename Dynamics::Innovations (&innovations)[NodeCapacity]
    ) {
        Dynamics::template draw_coupled<NodeCapacity>(
            random, prepared, node_count, innovations
        );
    }) {
        typename Dynamics::Innovations innovations[NodeCapacity];
        Dynamics::template draw_coupled<NodeCapacity>(
            random, prepared, node_count, innovations
        );
        #pragma unroll
        for (unsigned int node = 0U; node < NodeCapacity; ++node) {
            if (node < node_count) apply(node, innovations[node]);
        }
    } else {
        const auto innovation = [&] {
            if constexpr (requires {
                Dynamics::draw(random, prepared[0U]);
            }) {
                return Dynamics::draw(random, prepared[0U]);
            } else {
                return Dynamics::draw(random);
            }
        }();
        #pragma unroll
        for (unsigned int node = 0U; node < NodeCapacity; ++node) {
            if (node < node_count) apply(node, innovation);
        }
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
