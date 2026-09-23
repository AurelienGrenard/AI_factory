// Included device definitions of the Schobel--Zhu common-innovation adapter.
#pragma once

#include "model/equity/markovian/schobel_zhu/price_gradients/coupled_dynamics.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::schobel_zhu::price_gradients {

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters,
    float dt
) {
    return prepare_model(parameters, dt);
}

__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(
    const Prepared& prepared
) {
    return initial_state(prepared);
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
        innovations.ou_normal,
        innovations.increment_residual_normal,
        innovations.asset_residual_normal,
        state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::schobel_zhu::price_gradients
