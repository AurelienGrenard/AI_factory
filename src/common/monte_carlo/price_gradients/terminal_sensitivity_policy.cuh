// Shared terminal-node preparation and payoff contraction for MC sensitivities.
#pragma once

#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/philox.cuh"
#include "common/price_gradients/device_preparation.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/reconstruction.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

struct TerminalSpotObservation {
    struct State {
        float unscaled_spot;
        float scale;
    };

    __device__ __forceinline__ static float spot(const State& state) {
        return state.unscaled_spot * state.scale;
    }
};

template<typename Dynamics, typename ProductPolicy, typename Preparation>
struct TerminalNodePolicy {
    using Scenario = typename Preparation::Scenario;
    using PreparedDynamics = typename Dynamics::Prepared;
    using PreparedProduct = typename ProductPolicy::PreparedProduct;
    using Observation = TerminalSpotObservation;
    using NodeValue = float;

    struct Metadata {
        PreparedProduct product;
        float spot_scale;
    };

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Scenario& scenario,
        pg::TimeConfiguration time
    ) {
        auto model = scenario.model;
        model.spot = scenario.simulation_spot;
        const float horizon = Dynamics::kExactTerminal
            ? scenario.maturity_years
            : time.dt;
        return Dynamics::prepare(model, horizon);
    }

    __device__ __forceinline__ static PreparedProduct prepare_product(
        const Scenario& scenario,
        pg::TimeConfiguration time
    ) {
        return ProductPolicy::prepare_product(
            scenario.model,
            scenario.product,
            {
                static_cast<float>(time.simulation_steps_per_day) * time.dt,
                scenario.maturity_years,
            }
        );
    }

    __device__ __forceinline__ static Metadata prepare_metadata(
        const Scenario& scenario,
        pg::TimeConfiguration time
    ) {
        return {
            prepare_product(scenario, time),
            scenario.spot_scale,
        };
    }

    __device__ __forceinline__ static NodeValue observe(
        const typename Dynamics::State& state
    ) {
        return Dynamics::spot(state);
    }

    __device__ __forceinline__ static float payoff(
        const Metadata& metadata,
        NodeValue value
    ) {
        const auto handler = ProductPolicy::make_handler(metadata.product);
        return ProductPolicy::template finalize<Observation>(
            metadata.product,
            {value, metadata.spot_scale},
            handler
        );
    }

    __device__ __forceinline__ static float centered_first(
        const Metadata& first,
        const Metadata& second,
        NodeValue first_value,
        NodeValue second_value,
        float represented_width
    ) {
        return ProductPolicy::template centered_difference<Observation>(
            first.product,
            second.product,
            {first_value, first.spot_scale},
            {second_value, second.spot_scale},
            represented_width
        );
    }
};

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation,
    typename = void
>
struct TerminalNodePolicySelector {
    using type = TerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
};

template<typename Dynamics, typename ProductPolicy, typename Preparation>
struct TerminalNodePolicySelector<
    Dynamics,
    ProductPolicy,
    Preparation,
    std::void_t<typename ProductPolicy::template SensitivityNodePolicy<
        Dynamics, Preparation
    >>
> {
    using type = typename ProductPolicy::template SensitivityNodePolicy<
        Dynamics, Preparation
    >;
};

template<typename Dynamics, typename ProductPolicy, typename Preparation>
using SelectedTerminalNodePolicy = typename TerminalNodePolicySelector<
    Dynamics, ProductPolicy, Preparation
>::type;

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation>
struct TerminalSensitivityPolicy {
    static_assert(pg::requests_second_v<Orders>);
    static constexpr std::size_t kNodeCapacity = 4U;
    using NodePolicy =
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
    using Scenario = typename NodePolicy::Scenario;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;
    using Nodes = pg::SensitivityNodes<Scenario, kNodeCapacity>;
    using Metadata = typename NodePolicy::Metadata;
    using NodeValue = typename NodePolicy::NodeValue;

    struct PreparedRow {
        typename Dynamics::Prepared dynamics[kNodeCapacity];
        Metadata metadata[kNodeCapacity];
        const Nodes* nodes;
        const Stencil* stencil;
        std::uint32_t maximum_steps;
        std::uint8_t node_count;

        __device__ __forceinline__ const Scenario& input(
            unsigned int node
        ) const {
            return (*nodes)[node];
        }
    };

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time
    ) {
        PreparedRow row{};
        row.nodes = nodes;
        row.stencil = stencil;
        row.node_count = static_cast<std::uint8_t>(
            pg::active_node_count(*stencil)
        );
        #pragma unroll
        for (unsigned int node = 0U; node < kNodeCapacity; ++node) {
            if (node >= row.node_count) continue;
            const auto& input = row.input(node);
            row.dynamics[node] = NodePolicy::prepare_dynamics(input, time);
            row.metadata[node] = NodePolicy::prepare_metadata(input, time);
            row.maximum_steps = max(row.maximum_steps, input.step_count);
        }
        return row;
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
        typename Dynamics::State states[kNodeCapacity];
        #pragma unroll
        for (unsigned int node = 0U; node < kNodeCapacity; ++node) {
            if (node < row.node_count) {
                states[node] = Dynamics::initial(row.dynamics[node]);
            }
        }
        typename Dynamics::RandomContext random(key, path);
        simulate_coupled_terminal_nodes<kNodeCapacity, Dynamics>(
            random,
            row.dynamics,
            row.node_count,
            row.maximum_steps,
            [&](unsigned int node) {
                return node == 0U || !row.input(node).reuse_central;
            },
            [&](unsigned int node) { return row.input(node).step_count; },
            [&](unsigned int node) {
                return row.input(node).normal_weights;
            },
            [&](unsigned int node) -> typename Dynamics::State& {
                return states[node];
            }
        );

        pg::SensitivityValues<kNodeCapacity> values{};
        NodeValue terminal_values[kNodeCapacity]{};
        terminal_values[0U] = NodePolicy::observe(states[0U]);
        values[0U] = NodePolicy::payoff(
            row.metadata[0U], terminal_values[0U]
        );
        #pragma unroll
        for (unsigned int node = 1U; node < kNodeCapacity; ++node) {
            if (node < row.node_count) {
                const auto& terminal_state = row.input(node).reuse_central
                    ? states[0U]
                    : states[node];
                terminal_values[node] = NodePolicy::observe(terminal_state);
                values[node] = NodePolicy::payoff(
                    row.metadata[node], terminal_values[node]
                );
            }
        }
        auto sensitivity = pg::reconstruct_sensitivity<Orders>(
            *row.stencil, values
        );
        if constexpr (pg::requests_first_v<Orders>) {
            if (row.stencil->kind == pg::StencilKind::centered) {
                sensitivity.first = NodePolicy::centered_first(
                    row.metadata[1U],
                    row.metadata[2U],
                    terminal_values[1U],
                    terminal_values[2U],
                    row.stencil->represented_width
                );
            }
        }
        return {values[0U], sensitivity.first, sensitivity.second};
    }
};

template<pg::SensitivityOrders Orders, typename Preparation>
__device__ __forceinline__ bool build_terminal_sensitivity_task_from_central(
    const typename Preparation::Scenario& central,
    pg::SensitivitySpec<typename Preparation::Parameter> sensitivity,
    pg::TimeConfiguration time,
    pg::SensitivityTask<
        typename Preparation::Scenario,
        pg::SensitivityTraits<Orders>::node_capacity
    >& task,
    int& error
) {
    return preparation::build_sensitivity_task<Orders, Preparation>(
        central, sensitivity, time, task, error
    );
}

template<
    pg::SensitivityOrders Orders,
    typename Inputs,
    typename Preparation>
__device__ __forceinline__ bool build_terminal_sensitivity_task(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t row,
    std::size_t sensitivity_index,
    pg::SensitivityTask<
        typename Preparation::Scenario,
        pg::SensitivityTraits<Orders>::node_capacity
    >& task,
    int& error
) {
    typename Preparation::Scenario central{};
    if (!inputs.make_central(row, plan, central)) {
        error = preparation::invalid_central;
        return false;
    }
    return build_terminal_sensitivity_task_from_central<
        Orders, Preparation
    >(
        central,
        inputs.sensitivities[sensitivity_index],
        plan.time,
        task,
        error
    );
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
