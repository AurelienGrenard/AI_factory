// Included device definitions of the Heston original-order common-innovation adapter.
#pragma once
#include "model/equity/markovian/heston/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/heston/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::heston::price_gradients {
__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(const ModelParameters& p, float dt) {
    return heston::prepare_model(p, dt);
}
__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& p, const Innovations& z, const float*, State& state
) {
    heston::one_step_transition(p, z.variance_normal, z.variance_uniform, z.stock_normal, state);
}
__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(const Prepared& p) {
    return heston::initial_state(p);
}
__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(RandomContext& random) {
    const float variance = philox::next_normal(random.uniforms, random.normals);
    const float stock = philox::next_normal(random.uniforms, random.normals);
    const float uniform = random.uniforms.next();
    return {variance, uniform, stock};
}
__device__ __forceinline__ float CoupledDynamics::spot(const State& state) { return DynamicsPolicy::spot(state); }
}  // namespace ai_factory::workbench::model::equity::heston::price_gradients
