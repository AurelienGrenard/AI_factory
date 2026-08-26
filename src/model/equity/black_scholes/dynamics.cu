// Exact Black-Scholes preparation and path simulation.
#pragma once

#include "model/equity/black_scholes/dynamics.cuh"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::model::equity::black_scholes {

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float variance = parameters.volatility * parameters.volatility;
    return {
        logf(parameters.spot),
        parameters.risk_free_rate - parameters.dividend_yield
            - 0.5f * variance,
        parameters.volatility,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    return {
        prepared_model.drift_rate * delta_t,
        prepared_model.volatility * sqrtf(delta_t),
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return {prepared_model.initial_log_spot};
}

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& prepared_transition,
    float brownian_normal,
    State& state
) {
    state.log_spot = fmaf(
        prepared_transition.diffusion_standard_deviation,
        brownian_normal,
        state.log_spot + prepared_transition.drift
    );
}

// ==================== Model-specific implementation =======================

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel&,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    one_step_transition(
        prepared_transition, philox::next_normal(uniforms, normal_cache), state
    );
}

}  // namespace

// ======================== Common equity dynamics =========================

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    const PreparedModel prepared_model =
        DynamicsPolicy::prepare_model(parameters);
    return {
        prepared_model,
        DynamicsPolicy::prepare_transition(prepared_model, delta_t),
    };
}

__device__ __forceinline__ DynamicsPolicy::PreparedModel
DynamicsPolicy::prepare_model(const Parameters& parameters) {
    return black_scholes::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    return black_scholes::prepare_transition(prepared_model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return black_scholes::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& prepared_model) {
    return black_scholes::initial_state(prepared_model);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    DynamicsPolicy::simulate_one_step(
        dynamics.model,
        dynamics.transition,
        random,
        state
    );
}

__device__ __forceinline__ void DynamicsPolicy::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        DynamicsPolicy::simulate_one_step(dynamics, random, state);
    }
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    RandomContext& random,
    State& state
) {
    black_scholes::simulate_one_step(
        prepared_model,
        prepared_transition,
        random.uniforms,
        random.normals,
        state
    );
}

__device__ __forceinline__ float DynamicsPolicy::spot(
    const State& state
) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float DynamicsPolicy::log_spot(
    const State& state
) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::model::equity::black_scholes
