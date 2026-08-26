#pragma once

#include "model/equity/variance_gamma/dynamics.cuh"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::model::equity::variance_gamma {

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float sigma2 = parameters.sigma * parameters.sigma;
    const float martingale_argument = 1.0f
        - parameters.theta * parameters.nu
        - 0.5f * sigma2 * parameters.nu;
    const float martingale_correction =
        logf(martingale_argument) / parameters.nu;
    return {
        logf(parameters.spot),
        parameters.risk_free_rate - parameters.dividend_yield
            + martingale_correction,
        1.0f / parameters.nu,
        parameters.nu,
        parameters.theta,
        parameters.sigma,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    return {
        prepared_model.drift_rate * delta_t,
        delta_t * prepared_model.inverse_nu,
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return {prepared_model.initial_log_spot};
}

__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    float gamma_increment,
    float brownian_normal,
    State& state
) {
    const float brownian_increment =
        prepared_model.sigma * sqrtf(gamma_increment) * brownian_normal;
    state.log_spot += fmaf(
        prepared_model.theta,
        gamma_increment,
        prepared_transition.drift + brownian_increment
    );
}

// ==================== Model-specific implementation =======================

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    const float gamma_increment = philox::marsaglia_tsang_gamma(
        uniforms,
        normal_cache,
        prepared_transition.gamma_shape,
        prepared_model.nu
    );
    const float brownian_normal = philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        prepared_model, prepared_transition, gamma_increment, brownian_normal, state
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
    return variance_gamma::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    return variance_gamma::prepare_transition(prepared_model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return variance_gamma::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& prepared_model) {
    return variance_gamma::initial_state(prepared_model);
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
    variance_gamma::simulate_one_step(
        prepared_model,
        prepared_transition,
        random.uniforms,
        random.normals,
        state
    );
}

__device__ __forceinline__ float DynamicsPolicy::spot(const State& state) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float DynamicsPolicy::log_spot(
    const State& state
) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::model::equity::variance_gamma
