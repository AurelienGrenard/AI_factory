// Device transition definitions for SABR CRN scenarios with normalized initial volatility.
#pragma once

#include "model/equity/markovian/sabr/price_delta_dynamics.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::sabr {

__device__ __forceinline__ PriceDeltaDynamics::PreparedDynamics
PriceDeltaDynamics::prepare_dynamics(const Parameters& parameters, float dt) {
    auto lower = parameters.model;
    auto upper = parameters.model;
    lower.spot = parameters.lower_spot;
    upper.spot = parameters.upper_spot;
    return {DynamicsPolicy::prepare_dynamics(parameters.model, dt),
            DynamicsPolicy::prepare_dynamics(lower, dt),
            DynamicsPolicy::prepare_dynamics(upper, dt)};
}

__device__ __forceinline__ PriceDeltaDynamics::State
PriceDeltaDynamics::initial_state(const PreparedDynamics& prepared) {
    return {DynamicsPolicy::initial_state(prepared.central),
            DynamicsPolicy::initial_state(prepared.lower),
            DynamicsPolicy::initial_state(prepared.upper)};
}

__device__ __forceinline__ void PriceDeltaDynamics::advance(
    const PreparedDynamics& prepared, std::uint32_t step_count,
    RandomContext& random, State& state
) {
    for (std::uint32_t step = 0; step < step_count; ++step) {
        const float alpha_normal = philox::next_normal(random.uniforms, random.normals);
        const float residual_normal = philox::next_normal(random.uniforms, random.normals);
        one_step_transition(prepared.central, alpha_normal, residual_normal, state.central);
        one_step_transition(prepared.lower, alpha_normal, residual_normal, state.lower);
        one_step_transition(prepared.upper, alpha_normal, residual_normal, state.upper);
    }
}

template<unsigned int Scenario>
__device__ __forceinline__
::ai_factory::workbench::equity::price_delta::SpotObservation
PriceDeltaDynamics::observe(const State& state) {
    static_assert(Scenario < 3U);
    const auto& selected = [&]() -> const sabr::State& {
        if constexpr (Scenario == 0U) return state.central;
        else if constexpr (Scenario == 1U) return state.lower;
        else return state.upper;
    }();
    return {DynamicsPolicy::spot(selected), DynamicsPolicy::log_spot(selected)};
}

}  // namespace ai_factory::workbench::model::equity::sabr
