#pragma once

#include "model/equity/markovian/merton/dynamics.cuh"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::model::equity::merton {

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float jump_variance =
        parameters.jump_log_volatility * parameters.jump_log_volatility;
    const float jump_martingale = expf(
        parameters.jump_log_mean + 0.5f * jump_variance
    ) - 1.0f;
    const float diffusion_variance =
        parameters.volatility * parameters.volatility;
    return {
        logf(parameters.spot),
        parameters.risk_free_rate
            - parameters.dividend_yield
            - parameters.jump_intensity * jump_martingale
            - 0.5f * diffusion_variance,
        parameters.volatility,
        parameters.jump_intensity,
        parameters.jump_log_mean,
        parameters.jump_log_volatility,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    const float poisson_mean = prepared_model.jump_intensity * delta_t;
    return {
        prepared_model.drift_rate * delta_t,
        prepared_model.volatility * sqrtf(delta_t),
        poisson_mean,
        expf(-poisson_mean),
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
    std::uint32_t jump_count,
    float diffusion_normal,
    float jump_normal,
    State& state
) {
    const float count = static_cast<float>(jump_count);
    const float jump_log_sum = count * prepared_model.jump_log_mean
        + prepared_model.jump_log_volatility * sqrtf(count) * jump_normal;
    state.log_spot += prepared_transition.drift
        + prepared_transition.diffusion_standard_deviation * diffusion_normal
        + jump_log_sum;
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
    constexpr float kPoissonInversionThreshold = 10.0f;
    const std::uint32_t jump_count =
        prepared_transition.poisson_mean < kPoissonInversionThreshold
        ? philox::poisson_from_uniform(
            uniforms.next(),
            prepared_transition.poisson_mean,
            prepared_transition.zero_jump_probability
        )
        : philox::poisson_from_uniform_sequence(
            uniforms,
            prepared_transition.poisson_mean
        );
    const float diffusion_normal = philox::next_normal(uniforms, normal_cache);
    const float jump_normal = jump_count == 0U
        ? 0.0f
        : philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        prepared_model,
        prepared_transition,
        jump_count,
        diffusion_normal,
        jump_normal,
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
    return merton::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    return merton::prepare_transition(prepared_model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return merton::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& prepared_model) {
    return merton::initial_state(prepared_model);
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
    merton::simulate_one_step(
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

}  // namespace ai_factory::workbench::model::equity::merton
