// Included device definitions of the Black-Scholes common-innovation adapter.
#pragma once
#include "model/equity/markovian/black_scholes/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes::price_gradients {
__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(const ModelParameters& p, float years) {
    return DynamicsPolicy::prepare_dynamics(p, years);
}
__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(const Prepared& p) {
    return DynamicsPolicy::initial_state(p);
}
__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(RandomContext& random) {
    // Central endpoint first; bridge normals never change its Philox coordinate.
    const float central = philox::next_normal(random.uniforms, random.normals);
    const float first = philox::next_normal(random.uniforms, random.normals);
    const float second = philox::next_normal(random.uniforms, random.normals);
    return {{central, first, second}};
}
__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& p, const Innovations& z, const float* weights, State& state
) {
    const float normal = fmaf(weights[2], z.normals[2], fmaf(weights[1], z.normals[1], weights[0] * z.normals[0]));
    black_scholes::one_step_transition(p.transition, normal, state);
}
__device__ __forceinline__ float CoupledDynamics::spot(const State& state) { return DynamicsPolicy::spot(state); }
}  // namespace ai_factory::workbench::model::equity::black_scholes::price_gradients
