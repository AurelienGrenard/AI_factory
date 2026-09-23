// Included device definitions of the Stein--Stein common-innovation adapter.
#pragma once

#include "model/equity/markovian/stein_stein/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/stein_stein/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::stein_stein::price_gradients {

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters,
    float dt
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
    return {
        philox::next_normal(random.uniforms, random.normals),
        philox::next_normal(random.uniforms, random.normals),
        philox::next_normal(random.uniforms, random.normals),
    };
}

__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared,
    const Innovations& innovations,
    const float*,
    State& state
) {
    one_step_transition(
        prepared,
        innovations.endpoint_normal,
        innovations.increment_residual_normal,
        innovations.asset_residual_normal,
        state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::stein_stein::price_gradients
