// Exact Ornstein-Uhlenbeck state and optional state-integral simulation.
#pragma once

#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cuh"
#include "model/fixed_income/common/mean_reverting_gaussian.cuh"

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

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
    return {moments.state_loading, moments.variance};
}

__device__ __forceinline__ PreparedModel prepare_model(
    const ProcessParameters& parameters
) {
    return {
        parameters.mean_reversion,
        parameters.volatility * parameters.volatility,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    const float one_minus_decay = -expm1f(-model.mean_reversion * delta_t);
    const float decay = 1.0f - one_minus_decay;
    const float variance = mean_reverting_gaussian::state_variance_from_decay(
        model.mean_reversion,
        model.volatility_squared,
        decay,
        one_minus_decay
    );
    return {decay, sqrtf(fmaxf(variance, 0.0f))};
}

__device__ __forceinline__ void prepare_calendar(
    const PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
) {
    for (std::uint32_t interval = 0U; interval < interval_count; ++interval) {
        transitions[interval] = ornstein_uhlenbeck::prepare_transition(
            model, static_cast<float>(interval_steps[interval]) * delta_t
        );
    }
}

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float state_normal,
    float& state
) {
    state = fmaf(
        transition.decay,
        state,
        transition.state_standard_deviation * state_normal
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel&,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    float& state
) {
    one_step_transition(
        transition, philox::next_normal(uniforms, normals), state
    );
}

}  // namespace

__device__ __forceinline__ float simulate_terminal_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path
) {
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    simulate_one_step(model, transition, uniforms, normals, initial_state);
    return initial_state;
}

__device__ __forceinline__ float simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states
) {
    float state = initial_state;
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    for (std::uint32_t observation = 0U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, transitions[observation], uniforms, normals, state
        );
        observed_states[
            static_cast<std::size_t>(observation) * observation_stride
        ] = state;
    }
    simulate_one_step(
        model, transitions[observation_count - 1U], uniforms, normals, state
    );
    return state;
}

__device__ __forceinline__ float simulate_on_regular_grid(
    const PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states
) {
    float state = initial_state;
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    simulate_one_step(
        model, initial_stub_transition, uniforms, normals, state
    );
    if (observation_count == 1U) return state;
    observed_states[0U] = state;
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(model, regular_transition, uniforms, normals, state);
        observed_states[
            static_cast<std::size_t>(observation) * observation_stride
        ] = state;
    }
    simulate_one_step(model, regular_transition, uniforms, normals, state);
    return state;
}

namespace joint {

__device__ __forceinline__ PreparedTransition prepare_transition(
    const ornstein_uhlenbeck::PreparedModel& model,
    float delta_t
) {
    const float a = model.mean_reversion;
    const float one_minus_decay = -expm1f(-a * delta_t);
    const float decay = 1.0f - one_minus_decay;
    const float state_variance =
        mean_reverting_gaussian::state_variance_from_decay(
            a, model.volatility_squared, decay, one_minus_decay
        );
    const float state_standard_deviation = sqrtf(fmaxf(state_variance, 0.0f));
    const float covariance =
        mean_reverting_gaussian::state_integral_covariance_from_decay(
            a, model.volatility_squared, one_minus_decay
        );
    const float integral_state_normal_loading =
        state_standard_deviation > 0.0f
        ? covariance / state_standard_deviation
        : 0.0f;
    const float independent_variance = fmaxf(
        mean_reverting_gaussian::integral_variance_from_decay(
            a,
            model.volatility_squared,
            delta_t,
            decay,
            one_minus_decay
        ) - integral_state_normal_loading * integral_state_normal_loading,
        0.0f
    );
    return {
        decay,
        state_standard_deviation,
        one_minus_decay / a,
        integral_state_normal_loading,
        sqrtf(independent_variance),
    };
}

__device__ __forceinline__ void prepare_calendar(
    const ornstein_uhlenbeck::PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
) {
    for (std::uint32_t interval = 0U; interval < interval_count; ++interval) {
        transitions[interval] = ornstein_uhlenbeck::joint::prepare_transition(
            model, static_cast<float>(interval_steps[interval]) * delta_t
        );
    }
}

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float state_normal,
    float integral_normal,
    State& state
) {
    const float previous_state = state.state;
    const float state_noise =
        transition.state_standard_deviation * state_normal;
    const float integral_noise = fmaf(
        transition.integral_state_normal_loading,
        state_normal,
        transition.integral_independent_standard_deviation * integral_normal
    );
    state.state = fmaf(transition.decay, previous_state, state_noise);
    state.state_integral = fmaf(
        transition.integral_state_loading,
        previous_state,
        state.state_integral + integral_noise
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const ornstein_uhlenbeck::PreparedModel&,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) {
    const float state_normal = philox::next_normal(uniforms, normals);
    const float integral_normal = philox::next_normal(uniforms, normals);
    one_step_transition(transition, state_normal, integral_normal, state);
}

}  // namespace

__device__ __forceinline__ State simulate_terminal_state(
    const ornstein_uhlenbeck::PreparedModel& model,
    const PreparedTransition& transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path
) {
    State state{initial_state, 0.0f};
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    simulate_one_step(model, transition, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ State simulate_on_calendar(
    const ornstein_uhlenbeck::PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
) {
    State state{initial_state, 0.0f};
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    for (std::uint32_t observation = 0U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, transitions[observation], uniforms, normals, state
        );
        const std::size_t output =
            static_cast<std::size_t>(observation) * observation_stride;
        observed_states[output] = state.state;
        observed_integrated_states[output] = state.state_integral;
    }
    simulate_one_step(
        model, transitions[observation_count - 1U], uniforms, normals, state
    );
    return state;
}

__device__ __forceinline__ State simulate_on_regular_grid(
    const ornstein_uhlenbeck::PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
) {
    State state{initial_state, 0.0f};
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    simulate_one_step(
        model, initial_stub_transition, uniforms, normals, state
    );
    if (observation_count == 1U) return state;
    observed_states[0U] = state.state;
    observed_integrated_states[0U] = state.state_integral;
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(model, regular_transition, uniforms, normals, state);
        const std::size_t output =
            static_cast<std::size_t>(observation) * observation_stride;
        observed_states[output] = state.state;
        observed_integrated_states[output] = state.state_integral;
    }
    simulate_one_step(model, regular_transition, uniforms, normals, state);
    return state;
}

}  // namespace joint
}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
