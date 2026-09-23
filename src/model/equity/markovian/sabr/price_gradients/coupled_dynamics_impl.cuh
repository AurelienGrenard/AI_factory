// Device implementation of SABR coupled dynamics for sensitivity nodes.
#pragma once

#include "model/equity/markovian/sabr/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::sabr::price_gradients {

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters, float dt
) {
    return DynamicsPolicy::prepare_dynamics(parameters, dt);
}

__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(
    const Prepared& prepared
) {
    return DynamicsPolicy::initial_state(prepared);
}

__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(
    RandomContext& random
) {
    const float alpha = philox::next_normal(random.uniforms, random.normals);
    const float residual = philox::next_normal(random.uniforms, random.normals);
    return {alpha, residual};
}

__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared,
    const Innovations& innovations,
    const float*,
    State& state
) {
    sabr::one_step_transition(
        prepared, innovations.alpha_normal, innovations.residual_normal, state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::sabr::price_gradients
