// Included device definitions of the Heston 3/2 common-innovation adapter.
#pragma once

#include "model/equity/markovian/heston_3_2/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/heston_3_2/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::heston_3_2::price_gradients {

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
        innovations.variance_normal,
        innovations.residual_normal,
        state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::heston_3_2::price_gradients
