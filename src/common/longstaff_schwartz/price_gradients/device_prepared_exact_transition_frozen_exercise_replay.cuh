// Exact-transition sensitivity nodes replayed over the canonical LSM calendar.
#pragma once

#include "common/longstaff_schwartz/price_gradients/device_prepared_frozen_exercise_nodes.cuh"
#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/philox.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

template<typename CoupledDynamics, std::size_t NodeCapacity>
struct DevicePreparedExactTransitionFrozenExerciseReplay {
    static_assert(NodeCapacity == 3U || NodeCapacity == 4U);
    static constexpr std::size_t kEndpointCapacity = NodeCapacity - 1U;
    using Dynamics = CoupledDynamics;
    using Metadata = FrozenExerciseNodeMetadata<NodeCapacity>;
    static constexpr bool kExactTransitionReplay = true;

    struct GraphPrepared {
        typename Dynamics::Prepared initial{};
        typename Dynamics::Prepared regular{};
    };

    template<typename Scenario>
    __device__ __forceinline__ static GraphPrepared prepare_graph_node(
        const Scenario& scenario,
        FrozenExerciseReplayTime time
    ) {
        auto model = scenario.model;
        model.spot = scenario.simulation_spot;
        return {
            Dynamics::prepare(model, time.first_exercise_time),
            Dynamics::prepare(model, time.exercise_interval),
        };
    }

    struct Prepared {
        // Exact-transition models have a maturity-aligned first stub and one
        // regular exercise interval. Each interval owns its prepared law.
        typename Dynamics::Prepared initial[NodeCapacity]{};
        typename Dynamics::Prepared regular[NodeCapacity]{};
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
            result.initial[node] = Dynamics::prepare(
                model, time.first_exercise_time
            );
            result.regular[node] = Dynamics::prepare(
                model, time.exercise_interval
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
        std::uint32_t,
        std::uint32_t
    ) {
        namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
        namespace pg = ::ai_factory::workbench::price_gradients;
        pg::SensitivityValues<NodeCapacity> spots{};
        typename Dynamics::State states[kEndpointCapacity]{};
        if (!initialize_frozen_exercise_replay<Dynamics>(
                prepared.nodes, prepared.initial, exercise, spots, states)) {
            return spots;
        }

        typename Dynamics::RandomContext random(
            key, static_cast<std::uint64_t>(path)
        );
        const auto replay_interval = [&](
            const typename Dynamics::Prepared (&interval)[NodeCapacity]
        ) {
            mcpg::simulate_coupled_equal_horizon_nodes<
                NodeCapacity,
                Dynamics
            >(
                random,
                interval,
                prepared.nodes.node_count,
                [&](unsigned int node) {
                    return node != 0U
                        && !prepared.nodes.reuse_central[node - 1U];
                },
                [&](unsigned int node) -> typename Dynamics::State& {
                    return states[node - 1U];
                }
            );
        };
        replay_interval(prepared.initial);
        for (std::uint32_t observation = 0U;
             observation < exercise.observation;
             ++observation) {
            replay_interval(prepared.regular);
        }
        finish_frozen_exercise_replay<Dynamics>(
            prepared.nodes, states, spots
        );
        return spots;
    }
};

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
