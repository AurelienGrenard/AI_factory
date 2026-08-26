#pragma once

#include "model/equity/normal_inverse_gaussian/dynamics.cuh"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float alpha2 = parameters.alpha * parameters.alpha;
    const float beta2 = parameters.beta * parameters.beta;
    const float gamma = sqrtf(alpha2 - beta2);
    const float beta_plus_one = parameters.beta + 1.0f;
    const float exponential_moment_root = sqrtf(
        alpha2 - beta_plus_one * beta_plus_one
    );
    const float martingale_correction = parameters.delta
        * (exponential_moment_root - gamma);
    return {
        logf(parameters.spot),
        parameters.risk_free_rate - parameters.dividend_yield
            + martingale_correction,
        parameters.delta,
        1.0f / gamma,
        parameters.beta,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    const float delta_dt = prepared_model.delta * delta_t;
    return {
        prepared_model.drift_rate * delta_t,
        delta_dt * prepared_model.inverse_gamma,
        delta_dt * delta_dt,
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
    float inverse_gaussian_increment,
    float brownian_normal,
    State& state
) {
    const float brownian_increment =
        sqrtf(inverse_gaussian_increment) * brownian_normal;
    state.log_spot += fmaf(
        prepared_model.beta,
        inverse_gaussian_increment,
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
    const float inverse_gaussian_increment =
        philox::michael_schucany_haas_inverse_gaussian(
            uniforms,
            normal_cache,
            prepared_transition.inverse_gaussian_mean,
            prepared_transition.inverse_gaussian_shape
        );
    const float brownian_normal = philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        prepared_model,
        prepared_transition,
        inverse_gaussian_increment,
        brownian_normal,
        state
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
    return normal_inverse_gaussian::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    return normal_inverse_gaussian::prepare_transition(prepared_model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return normal_inverse_gaussian::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& prepared_model) {
    return normal_inverse_gaussian::initial_state(prepared_model);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    DynamicsPolicy::simulate_one_step(
        dynamics.model, dynamics.transition, random, state
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
    normal_inverse_gaussian::simulate_one_step(
        prepared_model, prepared_transition, random.uniforms, random.normals, state
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

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
