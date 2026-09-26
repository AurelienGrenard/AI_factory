// Device definitions for the G2++ joint-state common-innovation adapter.
#pragma once

#include "model/fixed_income/g2_plus_plus/price_gradients/coupled_dynamics.cuh"
#include "model/fixed_income/g2/dynamics_impl.cuh"

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients {

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& model, float horizon
) {
    return g2_plus_plus::joint::DynamicsPolicy::prepare_dynamics(model, horizon);
}

__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(
    const Prepared& prepared
) {
    return g2_plus_plus::joint::DynamicsPolicy::initial_state(prepared);
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
    g2::joint::one_step_transition(
        prepared.transition,
        innovations.x,
        innovations.y,
        innovations.integral,
        state
    );
}

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients
