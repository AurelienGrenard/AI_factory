// Fixed-step row-local sensitivity nodes replayed to one frozen exercise date.
#pragma once

#include "common/longstaff_schwartz/price_gradients/device_prepared_frozen_exercise_nodes.cuh"
#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/philox.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

template<typename CoupledDynamics, std::size_t NodeCapacity>
struct DevicePreparedFixedStepFrozenExerciseReplay {
    static_assert(NodeCapacity == 3U || NodeCapacity == 4U);
    static constexpr std::size_t kEndpointCapacity = NodeCapacity - 1U;
    using Dynamics = CoupledDynamics;
    using Metadata = FrozenExerciseNodeMetadata<NodeCapacity>;

    struct Prepared {
        // The central dynamics anchors parameter-dependent random variables
        // even though the central state comes from the frozen exercise trace.
        typename Dynamics::Prepared dynamics[NodeCapacity]{};
        Metadata nodes{};
    };

    template<typename Scenario>
    __device__ __forceinline__ static Prepared prepare(
        const ::ai_factory::workbench::price_gradients::SensitivityNodes<
            Scenario,
            NodeCapacity
        >& nodes,
        std::size_t node_count,
        FrozenExerciseReplayTime time
    ) {
        Prepared result{};
        result.nodes = prepare_frozen_exercise_node_metadata(
            nodes, node_count
        );
        #pragma unroll
        for (std::size_t node = 0U; node < NodeCapacity; ++node) {
            if (node >= node_count) continue;
            auto model = nodes[node].model;
            model.spot = nodes[node].simulation_spot;
            result.dynamics[node] = Dynamics::prepare(
                model, time.numerical_step
            );
        }
        return result;
    }

    __device__ __forceinline__ static
    ::ai_factory::workbench::price_gradients::SensitivityValues<NodeCapacity>
    evaluate(
        const Prepared& prepared,
        longstaff_schwartz::FrozenExerciseTrace exercise,
        philox::PhiloxKey key,
        std::size_t path,
        std::uint32_t initial_transition_count,
        std::uint32_t transitions_per_exercise
    ) {
        namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
        namespace pg = ::ai_factory::workbench::price_gradients;
        pg::SensitivityValues<NodeCapacity> spots{};
        typename Dynamics::State states[kEndpointCapacity]{};
        if (!initialize_frozen_exercise_replay<Dynamics>(
                prepared.nodes, prepared.dynamics, exercise, spots, states)) {
            return spots;
        }

        typename Dynamics::RandomContext random(
            key, static_cast<std::uint64_t>(path)
        );
        const auto replay_interval = [&](std::uint32_t transition_count) {
            mcpg::simulate_coupled_terminal_nodes<NodeCapacity, Dynamics>(
                random,
                prepared.dynamics,
                prepared.nodes.node_count,
                transition_count,
                [&](unsigned int node) {
                    return node != 0U
                        && !prepared.nodes.reuse_central[node - 1U];
                },
                [&](unsigned int) { return transition_count; },
                [&](unsigned int) {
                    return static_cast<const float*>(nullptr);
                },
                [&](unsigned int node) -> typename Dynamics::State& {
                    return states[node - 1U];
                }
            );
        };
        replay_interval(initial_transition_count);
        for (std::uint32_t observation = 0U;
             observation < exercise.observation;
             ++observation) {
            replay_interval(transitions_per_exercise);
        }
        finish_frozen_exercise_replay<Dynamics>(
            prepared.nodes, states, spots
        );
        return spots;
    }
};

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
