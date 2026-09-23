// Included device definitions of the coupled exact NIG increment adapter.
#pragma once

#include "model/equity/markovian/normal_inverse_gaussian/price_gradients/coupled_dynamics.cuh"

#include "model/equity/markovian/normal_inverse_gaussian/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian::price_gradients {

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters,
    float years
) {
    const auto model = normal_inverse_gaussian::prepare_model(parameters);
    return {model, normal_inverse_gaussian::prepare_transition(model, years)};
}

__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(
    const Prepared& prepared
) {
    return normal_inverse_gaussian::initial_state(prepared.model);
}

__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(
    RandomContext& random,
    const Prepared& prepared
) {
    return normal_inverse_gaussian::draw_transition_innovations(
        prepared.transition, random.uniforms, random.normals
    );
}

template<std::size_t NodeCapacity>
__device__ __forceinline__ void CoupledDynamics::draw_coupled(
    RandomContext& random,
    const Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    Innovations (&innovations)[NodeCapacity]
) {
    // MSH has a fixed primitive budget: one normal and one selector. The
    // following normal is the Brownian innovation and may use the cached
    // Box-Muller mate, exactly as in the canonical dynamics.
    const float clock_normal = philox::next_normal(
        random.uniforms, random.normals
    );
    const float selector = random.uniforms.next();
    const float brownian = philox::next_normal(
        random.uniforms, random.normals
    );
    for (std::uint8_t node = 0U; node < node_count; ++node) {
        innovations[node].inverse_gaussian_increment =
            philox::michael_schucany_haas_inverse_gaussian_from_variates(
                clock_normal,
                selector,
                prepared[node].transition.inverse_gaussian_mean,
                prepared[node].transition.inverse_gaussian_shape
            );
        innovations[node].brownian_normal = brownian;
    }
}

__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared,
    const Innovations& innovations,
    const float*,
    State& state
) {
    normal_inverse_gaussian::one_step_transition(
        prepared.model,
        prepared.transition,
        innovations.inverse_gaussian_increment,
        innovations.brownian_normal,
        state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return normal_inverse_gaussian::DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian::price_gradients
