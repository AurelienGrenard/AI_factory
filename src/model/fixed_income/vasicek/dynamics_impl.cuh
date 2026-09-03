// Exact Vasicek state and optional state-integral simulation.
#pragma once

#include "model/fixed_income/vasicek/dynamics.cuh"
#include "common/fixed_income/mean_reverting_gaussian.cuh"

namespace ai_factory::workbench::model::fixed_income::vasicek {

namespace mean_reverting_gaussian =
    ::ai_factory::workbench::fixed_income::mean_reverting_gaussian;

__device__ __forceinline__ float integral_state_loading(
    float mean_reversion,
    float delta
) {
    return mean_reverting_gaussian::integral_state_loading(
        mean_reversion, delta
    );
}

__device__ __forceinline__ float integral_variance(
    const ProcessParameters& parameters,
    float delta
) {
    return mean_reverting_gaussian::integral_variance(
        parameters.mean_reversion, parameters.volatility, delta
    );
}

__device__ __forceinline__ IntegralMoments integral_moments(
    const ProcessParameters& parameters,
    float delta
) {
    const mean_reverting_gaussian::IntegralMoments moments =
        mean_reverting_gaussian::integral_moments(
            parameters.mean_reversion, parameters.volatility, delta
        );
    return {
        moments.state_loading,
        parameters.long_term_mean * (delta - moments.state_loading),
        moments.variance,
    };
}

__device__ __forceinline__ PreparedModel prepare_model(
    const ProcessParameters& parameters
) {
    return {
        parameters.mean_reversion,
        parameters.long_term_mean,
        parameters.volatility * parameters.volatility,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    const float one_minus_decay = -expm1f(-prepared_model.mean_reversion * delta_t);
    const float state_decay = 1.0f - one_minus_decay;
    const float variance = mean_reverting_gaussian::state_variance_from_decay(
        prepared_model.mean_reversion,
        prepared_model.volatility_squared,
        state_decay,
        one_minus_decay
    );
    return {
        state_decay,
        prepared_model.long_term_mean * one_minus_decay,
        sqrtf(fmaxf(variance, 0.0f)),
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return prepared_model.initial_state;
}

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& prepared_transition,
    float state_normal,
    State& state
) {
    const float noise = prepared_transition.state_standard_deviation * state_normal;
    state = fmaf(
        prepared_transition.state_decay, state, prepared_transition.state_mean_increment + noise
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel&,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    float& state
) {
    one_step_transition(
        prepared_transition, philox::next_normal(uniforms, normal_cache), state
    );
}

}  // namespace

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters, float delta_t
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
    PreparedModel prepared_model = vasicek::prepare_model(parameters.process);
    prepared_model.initial_state = parameters.initial_state;
    return prepared_model;
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& prepared_model, float delta_t
) {
    return vasicek::prepare_transition(prepared_model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return vasicek::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& prepared_model) {
    return vasicek::initial_state(prepared_model);
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
    vasicek::simulate_one_step(
        prepared_model, prepared_transition, random.uniforms, random.normals, state
    );
}

namespace joint {

__device__ __forceinline__ PreparedTransition prepare_transition(
    const vasicek::PreparedModel& prepared_model,
    float delta_t
) {
    const float a = prepared_model.mean_reversion;
    const float one_minus_decay = -expm1f(-a * delta_t);
    const float state_decay = 1.0f - one_minus_decay;
    const float state_variance =
        mean_reverting_gaussian::state_variance_from_decay(
            a, prepared_model.volatility_squared, state_decay, one_minus_decay
        );
    const float state_standard_deviation = sqrtf(fmaxf(state_variance, 0.0f));
    const float covariance =
        mean_reverting_gaussian::state_integral_covariance_from_decay(
            a, prepared_model.volatility_squared, one_minus_decay
        );
    const float integral_state_loading = one_minus_decay / a;
    const float integral_state_normal_loading =
        state_standard_deviation > 0.0f
        ? covariance / state_standard_deviation
        : 0.0f;
    const float independent_variance = fmaxf(
        mean_reverting_gaussian::integral_variance_from_decay(
            a,
            prepared_model.volatility_squared,
            delta_t,
            state_decay,
            one_minus_decay
        ) - integral_state_normal_loading * integral_state_normal_loading,
        0.0f
    );
    return {
        state_decay,
        prepared_model.long_term_mean * one_minus_decay,
        state_standard_deviation,
        integral_state_loading,
        prepared_model.long_term_mean * (delta_t - integral_state_loading),
        integral_state_normal_loading,
        sqrtf(independent_variance),
    };
}

__device__ __forceinline__ State initial_state(
    const vasicek::PreparedModel& prepared_model
) {
    return {vasicek::initial_state(prepared_model), 0.0f};
}

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& prepared_transition,
    float state_normal,
    float integral_normal,
    State& state
) {
    const float previous_state = state.state;
    const float state_noise =
        prepared_transition.state_standard_deviation * state_normal;
    const float integral_noise = fmaf(
        prepared_transition.integral_state_normal_loading,
        state_normal,
        prepared_transition.integral_independent_standard_deviation * integral_normal
    );
    state.state = fmaf(
        prepared_transition.state_decay,
        previous_state,
        prepared_transition.state_mean_increment + state_noise
    );
    state.state_integral = fmaf(
        prepared_transition.integral_state_loading,
        previous_state,
        state.state_integral
            + prepared_transition.integral_mean_increment
            + integral_noise
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const vasicek::PreparedModel&,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    const float state_normal = philox::next_normal(uniforms, normal_cache);
    const float integral_normal = philox::next_normal(uniforms, normal_cache);
    one_step_transition(prepared_transition, state_normal, integral_normal, state);
}

}  // namespace

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters, float delta_t
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
    return vasicek::DynamicsPolicy::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& prepared_model, float delta_t
) {
    return joint::prepare_transition(prepared_model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return DynamicsPolicy::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& prepared_model) {
    return joint::initial_state(prepared_model);
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
    joint::simulate_one_step(
        prepared_model, prepared_transition, random.uniforms, random.normals, state
    );
}

}  // namespace joint
}  // namespace ai_factory::workbench::model::fixed_income::vasicek
