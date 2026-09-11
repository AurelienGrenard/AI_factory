// Device transition definitions sharing one normal across three Milstein states.
#pragma once

#include "model/equity/markovian/cev/price_delta_dynamics.cuh"
#include "model/equity/markovian/cev/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::cev {

__device__ __forceinline__ PriceDeltaDynamics::PreparedDynamics
PriceDeltaDynamics::prepare_dynamics(const Parameters& parameters, float dt) {
    return {prepare_model(parameters.model, dt), parameters.lower_spot, parameters.upper_spot};
}

__device__ __forceinline__ PriceDeltaDynamics::State
PriceDeltaDynamics::initial_state(const PreparedDynamics& prepared) {
    return {cev::initial_state(prepared.model), {prepared.lower_spot}, {prepared.upper_spot}};
}

__device__ __forceinline__ void PriceDeltaDynamics::advance(
    const PreparedDynamics& prepared, std::uint32_t step_count,
    RandomContext& random, State& state
) {
    for (std::uint32_t step = 0; step < step_count; ++step) {
        const float normal = philox::next_normal(random.uniforms, random.normals);
        one_step_transition(prepared.model, normal, state.central);
        one_step_transition(prepared.model, normal, state.lower);
        one_step_transition(prepared.model, normal, state.upper);
    }
}

template<unsigned int Scenario>
__device__ __forceinline__
::ai_factory::workbench::equity::price_delta::SpotObservation
PriceDeltaDynamics::observe(const State& state) {
    static_assert(Scenario < 3U);
    if constexpr (Scenario == 0U)
        return {DynamicsPolicy::spot(state.central), DynamicsPolicy::log_spot(state.central)};
    else if constexpr (Scenario == 1U)
        return {DynamicsPolicy::spot(state.lower), DynamicsPolicy::log_spot(state.lower)};
    else
        return {DynamicsPolicy::spot(state.upper), DynamicsPolicy::log_spot(state.upper)};
}

}  // namespace ai_factory::workbench::model::equity::cev
