// Shared node preparation and coupled payoff evolution for path sensitivities.
#pragma once

#include "common/equity/path_product_policy.cuh"
#include "common/monte_carlo/price_gradients/coupled_node_innovations.cuh"
#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/monte_carlo/price_gradients/path_schedule.cuh"
#include "common/price_gradients/reconstruction.cuh"
#include "common/price_gradients/sensitivity_task.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace eq = ::ai_factory::workbench::equity;

template<typename StatePolicy>
struct ScaledPathObservation {
    struct State {
        typename StatePolicy::State state;
        float spot_scale;
        float log_spot_shift;
    };

    __device__ __forceinline__ static float spot(const State& state) {
        return StatePolicy::spot(state.state) * state.spot_scale;
    }

    __device__ __forceinline__ static float log_spot(const State& state)
    requires eq::LogSpotStatePolicy<StatePolicy> {
        return StatePolicy::log_spot(state.state) + state.log_spot_shift;
    }

    static constexpr bool kNativeLogSpot = [] {
        if constexpr (requires { StatePolicy::kNativeLogSpot; }) {
            return StatePolicy::kNativeLogSpot;
        }
        return false;
    }();
};

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule>
struct PathNodePolicy {
    using Scenario = typename Preparation::Scenario;
    using CanonicalDynamics = typename Schedule::Dynamics;
    using PreparedDynamics = typename Dynamics::Prepared;
    using PreparedProduct = typename ProductPolicy::PreparedProduct;
    using Handler = typename ProductPolicy::Handler;
    using Observation = ScaledPathObservation<CanonicalDynamics>;
    using NodeValue = float;

    struct Metadata {
        PreparedProduct product;
        float spot_scale;
        float log_spot_shift;
    };

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Scenario& scenario,
        float interval_years
    ) {
        auto model = scenario.model;
        model.spot = scenario.simulation_spot;
        return Dynamics::prepare(model, interval_years);
    }

    __device__ __forceinline__ static Metadata prepare_metadata(
        const Scenario& scenario,
        pg::TimeConfiguration time
    ) {
        const float day_fraction = path_day_fraction(time);
        return {
            ProductPolicy::prepare_product(
                scenario.model,
                scenario.product,
                {day_fraction, scenario.maturity_years}
            ),
            scenario.spot_scale,
            logf(scenario.spot_scale),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const Metadata& metadata
    ) {
        return ProductPolicy::make_handler(metadata.product);
    }

    __device__ __forceinline__ static typename Observation::State observation(
        const Metadata& metadata,
        const typename Dynamics::State& state
    ) {
        return {
            state,
            metadata.spot_scale,
            metadata.log_spot_shift,
        };
    }

    __device__ __forceinline__ static float coordinate(
        const Metadata& metadata,
        const typename Dynamics::State& state
    ) {
        const auto observed = observation(metadata, state);
        if constexpr (
            ProductPolicy::kObservationCoordinate
                == eq::ObservationCoordinate::spot
        ) {
            return Observation::spot(observed);
        } else {
            return Observation::log_spot(observed);
        }
    }

    __device__ __forceinline__ static float finalize(
        const Metadata& metadata,
        const typename Dynamics::State& terminal,
        const Handler& handler
    ) {
        return ProductPolicy::template finalize<Observation>(
            metadata.product,
            observation(metadata, terminal),
            handler
        );
    }

    __device__ __forceinline__ static float payoff(
        const Metadata&,
        NodeValue value
    ) {
        return value;
    }

    __device__ __forceinline__ static float centered_first(
        const Metadata&,
        const Metadata&,
        NodeValue first,
        NodeValue second,
        float represented_width
    ) {
        return (second - first) / represented_width;
    }
};

template<
    bool VariableTerminal,
    std::size_t NodeCapacity,
    typename Dynamics,
    typename Schedule>
struct PathRowScheduleStorage;

template<std::size_t NodeCapacity, typename Dynamics, typename Schedule>
struct PathRowScheduleStorage<false, NodeCapacity, Dynamics, Schedule> {
    using Traits = PathScheduleTraits<Schedule>;
    PreparedPathSchedule<Traits::kIntervalCapacity> schedule;
};

template<std::size_t NodeCapacity, typename Dynamics, typename Schedule>
struct PathRowScheduleStorage<true, NodeCapacity, Dynamics, Schedule> {
    using Traits = PathScheduleTraits<Schedule>;
    PreparedTerminalPathSchedule<Traits::kIntervalCapacity> schedule;
    typename Dynamics::Prepared terminal_dynamics[NodeCapacity];
    PreparedTerminalPathNode terminal_nodes[NodeCapacity];
    std::uint32_t maximum_terminal_transitions = 0U;
};

template<
    std::size_t NodeCapacity,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    bool VariableTerminal>
struct PreparedPathSensitivityRow
    : PathRowScheduleStorage<
          VariableTerminal, NodeCapacity, Dynamics, Schedule
      > {
    using CoupledDynamics = Dynamics;
    using NodePolicy =
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>;
    using Traits = PathScheduleTraits<Schedule>;
    using Scenario = typename Preparation::Scenario;
    using Stencil = pg::SensitivityStencil<NodeCapacity>;
    using Nodes = pg::SensitivityNodes<Scenario, NodeCapacity>;
    static constexpr std::size_t kNodeCapacity = NodeCapacity;
    static constexpr bool kVariableTerminal = VariableTerminal;

    typename Dynamics::Prepared
        dynamics[Traits::kIntervalCapacity][NodeCapacity];
    typename NodePolicy::Metadata metadata[NodeCapacity];
    const Nodes* nodes;
    const Stencil* stencil;
    std::uint8_t node_count;

    __device__ __forceinline__ const Scenario& input(
        unsigned int node
    ) const {
        return (*nodes)[node];
    }
};

template<
    std::size_t NodeCapacity,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    bool VariableTerminal>
__device__ __forceinline__ auto prepare_path_sensitivity_row(
    const pg::SensitivityNodes<
        typename Preparation::Scenario, NodeCapacity
    >* nodes,
    const pg::SensitivityStencil<NodeCapacity>* stencil,
    pg::TimeConfiguration time
) {
    using Row = PreparedPathSensitivityRow<
        NodeCapacity,
        Dynamics,
        ProductPolicy,
        Preparation,
        Schedule,
        VariableTerminal
    >;
    using NodePolicy = typename Row::NodePolicy;
    using Traits = typename Row::Traits;

    Row row{};
    row.nodes = nodes;
    row.stencil = stencil;
    row.node_count = static_cast<std::uint8_t>(
        pg::active_node_count(*stencil)
    );
    const auto calendar = ProductPolicy::calendar((*nodes)[0U].product);
    if constexpr (VariableTerminal) {
        row.schedule = prepare_terminal_path_schedule<Schedule>(
            calendar, time, (*nodes)[0U].central_step_count
        );
    } else {
        row.schedule = prepare_path_schedule<Schedule>(calendar, time);
    }
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node >= row.node_count) continue;
        row.metadata[node] = NodePolicy::prepare_metadata(
            row.input(node), time
        );
        #pragma unroll
        for (unsigned int interval = 0U;
             interval < Traits::kIntervalCapacity;
             ++interval) {
            const auto& central_schedule = [&]() -> const auto& {
                if constexpr (VariableTerminal) {
                    return row.schedule.central;
                } else {
                    return row.schedule;
                }
            }();
            const float horizon = Traits::kExactTransition
                ? central_schedule.interval_years[interval]
                : time.dt;
            row.dynamics[interval][node] =
                NodePolicy::prepare_dynamics(row.input(node), horizon);
        }
        if constexpr (VariableTerminal) {
            row.terminal_nodes[node] =
                prepare_terminal_path_node<Schedule>(
                    row.input(node), row.schedule
                );
            row.maximum_terminal_transitions = max(
                row.maximum_terminal_transitions,
                row.terminal_nodes[node].transition_count
            );
            const float horizon = Traits::kExactTransition
                ? row.terminal_nodes[node].interval_years
                : time.dt;
            row.terminal_dynamics[node] =
                NodePolicy::prepare_dynamics(row.input(node), horizon);
        }
    }
    return row;
}

template<bool IncludeCentral, typename Row>
__device__ __forceinline__ auto evaluate_path_nodes(
    const Row& row,
    philox::PhiloxKey key,
    std::size_t path
) -> pg::SensitivityValues<Row::kNodeCapacity> {
    using Dynamics = typename Row::CoupledDynamics;
    using NodePolicy = typename Row::NodePolicy;
    using Traits = typename Row::Traits;
    constexpr std::size_t NodeCapacity = Row::kNodeCapacity;
    constexpr unsigned int first_node = IncludeCentral ? 0U : 1U;

    typename Dynamics::State states[NodeCapacity]{};
    typename NodePolicy::Handler handlers[NodeCapacity]{};
    bool active[NodeCapacity]{};
    #pragma unroll
    for (unsigned int node = first_node; node < NodeCapacity; ++node) {
        if (node >= row.node_count) continue;
        if (node == 0U || !row.input(node).reuse_central) {
            states[node] = Dynamics::initial(row.dynamics[0U][node]);
        }
        handlers[node] = NodePolicy::make_handler(row.metadata[node]);
        const unsigned int state_node = row.input(node).reuse_central
            ? 0U : node;
        active[node] = handlers[node].on_initial_value(
            NodePolicy::coordinate(row.metadata[node], states[state_node])
        );
    }

    const auto simulates = [&](unsigned int node) {
        return (IncludeCentral || node != 0U)
            && (node == 0U || !row.input(node).reuse_central);
    };
    typename Dynamics::RandomContext random(key, path);
    const auto transition = [&](unsigned int interval) {
        draw_and_apply_node_innovations<NodeCapacity, Dynamics>(
            random,
            row.dynamics[interval],
            row.node_count,
            [&](unsigned int node, const auto& innovations) {
                if (simulates(node)) {
                    Dynamics::transition(
                        row.dynamics[interval][node],
                        innovations,
                        nullptr,
                        states[node]
                    );
                }
            }
        );
    };
    const auto equal_horizon_transition = [&](unsigned int interval) {
        simulate_coupled_equal_horizon_nodes<NodeCapacity, Dynamics>(
            random,
            row.dynamics[interval],
            row.node_count,
            simulates,
            [&](unsigned int node) -> typename Dynamics::State& {
                return states[node];
            }
        );
    };
    const auto observe = [&](std::uint32_t observation) {
        #pragma unroll
        for (unsigned int node = first_node;
             node < NodeCapacity;
             ++node) {
            if (node >= row.node_count || !active[node]) continue;
            const unsigned int state_node = row.input(node).reuse_central
                ? 0U : node;
            active[node] = handlers[node].on_observation(
                observation,
                NodePolicy::coordinate(
                    row.metadata[node], states[state_node]
                )
            );
        }
    };

    if constexpr (!Row::kVariableTerminal) {
        if constexpr (Traits::kKind == PathScheduleKind::dense) {
            for (std::uint32_t observation = 0U;
                 observation < row.schedule.observation_count;
                 ++observation) {
                transition(0U);
                observe(observation);
            }
        } else {
            for (std::uint32_t observation = 0U;
                 observation < row.schedule.observation_count;
                 ++observation) {
                const unsigned int interval =
                    Traits::kKind == PathScheduleKind::calendar
                    ? observation
                    : 0U;
                const auto transition_count =
                    row.schedule.transition_counts[interval];
                if constexpr (Traits::kExactTransition) {
                    for (std::uint32_t step = 0U;
                         step < transition_count;
                         ++step) {
                        transition(interval);
                    }
                } else {
                    simulate_coupled_fixed_step_nodes<
                        NodeCapacity, Dynamics
                    >(
                        random,
                        row.dynamics[interval],
                        row.node_count,
                        transition_count,
                        simulates,
                        [&](unsigned int) { return transition_count; },
                        [&](unsigned int node)
                            -> typename Dynamics::State& {
                            return states[node];
                        }
                    );
                }
                observe(observation);
            }
        }
    } else {
        const auto& central_schedule = row.schedule.central;
        for (std::uint32_t observation = 0U;
             observation < row.schedule.prefix_observation_count;
             ++observation) {
            const unsigned int interval =
                Traits::kKind == PathScheduleKind::calendar
                ? observation
                : 0U;
            const auto transition_count =
                Traits::kKind == PathScheduleKind::dense
                ? 1U
                : central_schedule.transition_counts[interval];
            if constexpr (Traits::kExactTransition) {
                for (std::uint32_t step = 0U;
                     step < transition_count;
                     ++step) {
                    equal_horizon_transition(interval);
                }
            } else {
                simulate_coupled_fixed_step_nodes<
                    NodeCapacity, Dynamics
                >(
                    random,
                    row.dynamics[interval],
                    row.node_count,
                    transition_count,
                    simulates,
                    [&](unsigned int) { return transition_count; },
                    [&](unsigned int node)
                        -> typename Dynamics::State& {
                        return states[node];
                    }
                );
            }
            observe(observation);
        }

        if constexpr (Traits::kExactTransition) {
            if constexpr (requires {
                Dynamics::draw_maturity_coupled(random);
            }) {
                const auto innovations =
                    Dynamics::draw_maturity_coupled(random);
                #pragma unroll
                for (unsigned int node = 0U;
                     node < NodeCapacity;
                     ++node) {
                    if (node < row.node_count && simulates(node)) {
                        Dynamics::transition(
                            row.terminal_dynamics[node],
                            innovations,
                            row.input(node).normal_weights,
                            states[node]
                        );
                    }
                }
            } else {
                draw_and_apply_node_innovations<NodeCapacity, Dynamics>(
                    random,
                    row.terminal_dynamics,
                    row.node_count,
                    [&](unsigned int node, const auto& innovations) {
                        if (simulates(node)) {
                            Dynamics::transition(
                                row.terminal_dynamics[node],
                                innovations,
                                nullptr,
                                states[node]
                            );
                        }
                    }
                );
            }
        } else {
            simulate_coupled_fixed_step_nodes<NodeCapacity, Dynamics>(
                random,
                row.terminal_dynamics,
                row.node_count,
                row.maximum_terminal_transitions,
                simulates,
                [&](unsigned int node) {
                    return row.terminal_nodes[node].transition_count;
                },
                [&](unsigned int node) -> typename Dynamics::State& {
                    return states[node];
                }
            );
        }
        observe(row.schedule.prefix_observation_count);
    }

    pg::SensitivityValues<NodeCapacity> values{};
    #pragma unroll
    for (unsigned int node = first_node; node < NodeCapacity; ++node) {
        if (node >= row.node_count) continue;
        const unsigned int state_node = row.input(node).reuse_central
            ? 0U : node;
        values[node] = NodePolicy::finalize(
            row.metadata[node],
            states[state_node],
            handlers[node]
        );
    }
    return values;
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    bool VariableTerminal = false>
struct PathSensitivityPolicy {
    static_assert(pg::requests_second_v<Orders>);
    static constexpr std::size_t kNodeCapacity = 4U;
    using NodePolicy =
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>;
    using Scenario = typename Preparation::Scenario;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;
    using Nodes = pg::SensitivityNodes<Scenario, kNodeCapacity>;
    using PreparedRow = PreparedPathSensitivityRow<
        kNodeCapacity,
        Dynamics,
        ProductPolicy,
        Preparation,
        Schedule,
        VariableTerminal
    >;

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time
    ) {
        return prepare_path_sensitivity_row<
            kNodeCapacity,
            Dynamics,
            ProductPolicy,
            Preparation,
            Schedule,
            VariableTerminal
        >(nodes, stencil, time);
    }

    struct PathResult {
        float price;
        float first;
        float second;
    };

    __device__ __forceinline__ static PathResult evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        const auto values = evaluate_path_nodes<true>(
            row, key, path
        );
        const auto sensitivity = pg::reconstruct_sensitivity<Orders>(
            *row.stencil, values
        );
        return {values[0U], sensitivity.first, sensitivity.second};
    }
};

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename Schedule,
    bool VariableTerminal = false>
struct FirstPathSensitivityPolicy {
    static constexpr std::size_t kNodeCapacity = 3U;
    using NodePolicy =
        PathNodePolicy<Dynamics, ProductPolicy, Preparation, Schedule>;
    using Scenario = typename Preparation::Scenario;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;
    using Nodes = pg::SensitivityNodes<Scenario, kNodeCapacity>;
    using PreparedRow = PreparedPathSensitivityRow<
        kNodeCapacity,
        Dynamics,
        ProductPolicy,
        Preparation,
        Schedule,
        VariableTerminal
    >;

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time,
        bool
    ) {
        return prepare_path_sensitivity_row<
            kNodeCapacity,
            Dynamics,
            ProductPolicy,
            Preparation,
            Schedule,
            VariableTerminal
        >(nodes, stencil, time);
    }

    struct PathResult {
        float price;
        float sensitivity;
    };

    template<bool IncludeCentral, bool IncludePayoff>
    __device__ __forceinline__ static PathResult evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        static_assert(IncludeCentral || !IncludePayoff);
        const auto values = evaluate_path_nodes<IncludeCentral>(
            row, key, path
        );
        PathResult result{};
        if constexpr (IncludePayoff) result.price = values[0U];
        result.sensitivity = pg::reconstruct_first_sensitivity(
            *row.stencil, values
        );
        return result;
    }
};

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
