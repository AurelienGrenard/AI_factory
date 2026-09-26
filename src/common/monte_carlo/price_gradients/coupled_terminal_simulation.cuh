// Optional model-owned terminal coupling over the common node kernel.
#pragma once

#include "common/monte_carlo/price_gradients/coupled_node_innovations.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

// Most models consume one innovation per numerical step through the fallback.
// A model with an exactly aggregable independent component may instead expose
// simulate_coupled_terminal: it keeps the same node topology while owning the
// more efficient path algorithm used by its canonical terminal pricer.
template<
    std::size_t NodeCapacity,
    typename Dynamics,
    typename IsActive,
    typename StepCount,
    typename NormalWeights,
    typename State>
__device__ __forceinline__ void simulate_coupled_terminal_nodes(
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    std::uint16_t node_count,
    std::uint32_t maximum_steps,
    IsActive is_active,
    StepCount step_count,
    NormalWeights normal_weights,
    State state
) {
    if constexpr (requires {
        Dynamics::template simulate_coupled_terminal<NodeCapacity>(
            random,
            prepared,
            node_count,
            maximum_steps,
            is_active,
            step_count,
            state
        );
    }) {
        Dynamics::template simulate_coupled_terminal<NodeCapacity>(
            random,
            prepared,
            node_count,
            maximum_steps,
            is_active,
            step_count,
            state
        );
    } else if constexpr (Dynamics::kExactTerminal) {
        draw_and_apply_node_innovations<NodeCapacity, Dynamics>(
            random, prepared, node_count,
            [&](unsigned int node, const auto& innovations) {
                if (is_active(node)) {
                    Dynamics::transition(
                        prepared[node], innovations,
                        normal_weights(node), state(node)
                    );
                }
            }
        );
    } else {
        for (std::uint32_t step = 0U; step < maximum_steps; ++step) {
            draw_and_apply_node_innovations<NodeCapacity, Dynamics>(
                random, prepared, node_count,
                [&](unsigned int node, const auto& innovations) {
                    if (is_active(node) && step < step_count(node)) {
                        Dynamics::transition(
                            prepared[node], innovations, nullptr, state(node)
                        );
                    }
                }
            );
        }
    }
}

// Exact-transition exercise schedules replay every node over the same
// contractual interval. A terminal adapter may expose a cheaper shared draw
// for this case while retaining its richer maturity-coupling draw elsewhere.
template<
    std::size_t NodeCapacity,
    typename Dynamics,
    typename IsActive,
    typename State
>
__device__ __forceinline__ void simulate_coupled_equal_horizon_nodes(
    typename Dynamics::RandomContext& random,
    const typename Dynamics::Prepared (&prepared)[NodeCapacity],
    std::uint16_t node_count,
    IsActive is_active,
    State state
) {
    if constexpr (requires {
        Dynamics::draw_equal_horizon(random);
    }) {
        const auto innovations = Dynamics::draw_equal_horizon(random);
        #pragma unroll
        for (unsigned int node = 0U; node < NodeCapacity; ++node) {
            if (node < node_count && is_active(node)) {
                Dynamics::transition(
                    prepared[node], innovations, nullptr, state(node)
                );
            }
        }
    } else {
        simulate_coupled_terminal_nodes<NodeCapacity, Dynamics>(
            random,
            prepared,
            node_count,
            1U,
            is_active,
            [](unsigned int) { return 1U; },
            [](unsigned int) {
                return static_cast<const float*>(nullptr);
            },
            state
        );
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
