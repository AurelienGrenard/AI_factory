// Included device definitions of the CEV common-normal transition adapter.
#pragma once
#include "model/equity/markovian/cev/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/cev/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::cev::price_gradients {
__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(const ModelParameters& p, float dt) {
    return cev::prepare_model(p, dt);
}
__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(const Prepared& p) {
    return cev::initial_state(p);
}
__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(RandomContext& random) {
    return {philox::next_normal(random.uniforms, random.normals)};
}
__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& p, const Innovations& z, const float*, State& state
) {
    cev::one_step_transition(p, z.normal, state);
}
__device__ __forceinline__ float CoupledDynamics::spot(const State& state) { return DynamicsPolicy::spot(state); }
}  // namespace ai_factory::workbench::model::equity::cev::price_gradients
