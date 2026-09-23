// Included device definitions for the canonical Merton coupling interface.
#pragma once
#include "model/equity/markovian/merton/price_gradients/coupled_dynamics.cuh"
#include "common/monte_carlo/price_gradients/coupled_brownian_endpoints.cuh"
#include "common/monte_carlo/price_gradients/coupled_poisson_events.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::merton::price_gradients {
__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters, float years
) {
    const auto model = DynamicsPolicy::prepare_model(parameters);
    return {
        model,
        DynamicsPolicy::prepare_transition(model, years),
        years,
    };
}
__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(const Prepared& prepared) {
    return DynamicsPolicy::initial_state(prepared.model);
}
__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(
    RandomContext& random, const Prepared& central
) {
    return merton::draw_transition_innovations(central.transition, random);
}

namespace detail {

template<std::size_t NodeCapacity>
__device__ __forceinline__ void coupled_brownian_endpoints(
    philox::DomainUniformSequence& uniforms,
    philox::NormalPairCache& cache,
    const CoupledDynamics::Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    CoupledDynamics::Innovations (&innovations)[NodeCapacity]
) {
    ::ai_factory::workbench::monte_carlo::price_gradients::
        draw_coupled_brownian_normals<NodeCapacity>(
            uniforms, cache, node_count,
            [&](std::uint8_t node) { return prepared[node].horizon; },
            [&](std::uint8_t node, float normal) {
                innovations[node].diffusion_normal = normal;
            }
        );
}

template<std::size_t NodeCapacity>
__device__ __forceinline__ void coupled_jump_events(
    philox::DomainRandomContext& random,
    std::uint32_t step,
    const CoupledDynamics::Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    CoupledDynamics::Innovations (&innovations)[NodeCapacity]
) {
    float means[NodeCapacity]{};
    std::uint32_t counts[NodeCapacity]{};
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) {
            means[node] = prepared[node].transition.poisson_mean;
            innovations[node].jump_standard_normal_sum = 0.0f;
        }
    }
    ::ai_factory::workbench::monte_carlo::price_gradients::
        replay_coupled_poisson_events<
            NodeCapacity,
            ::ai_factory::workbench::monte_carlo::price_gradients::
                StandardNormalEventMarks
        >(
            random, step, means, node_count, counts,
            [&](std::uint8_t node, float mark) {
                innovations[node].jump_standard_normal_sum += mark;
            }
        );
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) innovations[node].jump_count = counts[node];
    }
}

}  // namespace detail

template<std::size_t NodeCapacity>
__device__ __forceinline__ void CoupledDynamics::draw_coupled(
    RandomContext& random,
    const Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    Innovations (&innovations)[NodeCapacity]
) {
    const auto step = random.next_step();
    auto diffusion_uniforms = random.source<1U>(step);
    philox::NormalPairCache diffusion_cache;
    detail::coupled_brownian_endpoints(
        diffusion_uniforms, diffusion_cache,
        prepared, node_count, innovations
    );
    detail::coupled_jump_events(
        random, step, prepared, node_count, innovations
    );
}

__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared, const Innovations& innovations, const float*, State& state
) {
    merton::one_step_transition(prepared.model, prepared.transition, innovations.jump_count,
        innovations.diffusion_normal,
        innovations.jump_standard_normal_sum, state);
}
__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return DynamicsPolicy::spot(state);
}
}  // namespace ai_factory::workbench::model::equity::merton::price_gradients
