// Included definitions for the canonical Merton variable-consumption coupling.
#pragma once
#include "model/equity/markovian/merton/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::merton::price_gradients {
__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters, float years
) {
    return DynamicsPolicy::prepare_dynamics(parameters, years);
}
__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(const Prepared& prepared) {
    return DynamicsPolicy::initial_state(prepared);
}
__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(
    RandomContext& random, const Prepared& central
) {
    return merton::draw_transition_innovations(central.transition, random);
}
__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared, const Innovations& innovations, const float*, State& state
) {
    merton::one_step_transition(prepared.model, prepared.transition, innovations.jump_count,
        innovations.diffusion_normal, innovations.jump_normal, state);
}
__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return DynamicsPolicy::spot(state);
}
}  // namespace ai_factory::workbench::model::equity::merton::price_gradients
