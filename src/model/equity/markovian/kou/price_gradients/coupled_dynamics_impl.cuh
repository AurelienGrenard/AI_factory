// Device definitions for the canonical Kou event-coupling adapter.
#pragma once

#include "model/equity/markovian/kou/price_gradients/coupled_dynamics.cuh"

#include "common/compound_poisson.cuh"
#include "common/monte_carlo/price_gradients/coupled_brownian_endpoints.cuh"
#include "common/monte_carlo/price_gradients/coupled_poisson_events.cuh"
#include "model/equity/markovian/kou/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::kou::price_gradients {

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters,
    float years
) {
    const auto model = kou::prepare_model(parameters);
    return {model, kou::prepare_transition(model, years), years};
}

__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(
    const Prepared& prepared
) {
    return kou::initial_state(prepared.model);
}

__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(
    RandomContext& random,
    const Prepared& prepared
) {
    return kou::draw_transition_innovations(
        prepared.model, prepared.transition, random
    );
}

template<std::size_t NodeCapacity>
__device__ __forceinline__ void CoupledDynamics::draw_coupled(
    RandomContext& random,
    const Prepared (&prepared)[NodeCapacity],
    std::uint16_t node_count,
    Innovations (&innovations)[NodeCapacity]
) {
    const auto step = random.next_step();
    auto diffusion_uniforms = random.source<
        compound_poisson::kContinuousSource
    >(step);
    philox::NormalPairCache diffusion_cache;
    ::ai_factory::workbench::monte_carlo::price_gradients::
        draw_coupled_brownian_normals<NodeCapacity>(
            diffusion_uniforms, diffusion_cache, node_count,
            [&](std::uint16_t node) { return prepared[node].horizon; },
            [&](std::uint16_t node, float normal) {
                innovations[node].diffusion_normal = normal;
            }
        );

    float means[NodeCapacity]{};
    std::uint32_t counts[NodeCapacity]{};
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) {
            means[node] = prepared[node].transition.poisson_mean;
            innovations[node].jump_log_sum = 0.0f;
        }
    }
    ::ai_factory::workbench::monte_carlo::price_gradients::
        replay_coupled_poisson_events<
            NodeCapacity,
            ::ai_factory::workbench::monte_carlo::price_gradients::
                UniformPairEventMarks
        >(
            random, step, means, node_count, counts,
            [&](
                std::uint8_t node,
                ::ai_factory::workbench::monte_carlo::price_gradients::
                    UniformPairEventMarks::Mark mark
            ) {
                const auto& model = prepared[node].model;
                const bool upward = mark.first < model.up_probability;
                const float magnitude = -logf(mark.second);
                innovations[node].jump_log_sum += upward
                    ? magnitude * model.inverse_positive_jump_rate
                    : -magnitude * model.inverse_negative_jump_rate;
            }
        );
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) innovations[node].jump_count = counts[node];
    }
}

__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared,
    const Innovations& innovations,
    const float*,
    State& state
) {
    kou::one_step_transition(
        prepared.transition,
        innovations.diffusion_normal,
        innovations.jump_log_sum,
        state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return kou::DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::kou::price_gradients
