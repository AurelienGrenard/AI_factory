// Exact CIR state simulation through the Poisson-Gamma representation.
#pragma once

#include "model/fixed_income/cir/dynamics.cuh"

namespace ai_factory::workbench::model::cir {

__device__ __forceinline__ PreparedModel prepare_model(
    const ProcessParameters& parameters
) {
    const float volatility_squared =
        parameters.volatility * parameters.volatility;
    return {
        parameters.mean_reversion,
        4.0f * parameters.mean_reversion * parameters.long_term_mean
            / volatility_squared,
        volatility_squared / (4.0f * parameters.mean_reversion),
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    const float one_minus_decay = -expm1f(-model.mean_reversion * delta_t);
    return {1.0f - one_minus_decay, model.scale_rate * one_minus_decay};
}

__device__ __forceinline__ void prepare_calendar(
    const PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
) {
    for (std::uint32_t interval = 0U; interval < interval_count; ++interval) {
        transitions[interval] = prepare_transition(
            model, static_cast<float>(interval_steps[interval]) * delta_t
        );
    }
}

__device__ __forceinline__ void one_step_transition(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    float& state
) {
    state = philox::scaled_noncentral_chi_square(
        uniforms,
        normal_cache,
        model.degrees_of_freedom,
        transition.decay * state / transition.scale,
        transition.scale
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    float& state
) {
    one_step_transition(model, transition, uniforms, normal_cache, state);
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
    philox::NormalPairCache normal_cache;
    simulate_one_step(
        model, transition, uniforms, normal_cache, initial_state
    );
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
    philox::NormalPairCache normal_cache;
    for (std::uint32_t observation = 0U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, transitions[observation], uniforms, normal_cache, state
        );
        observed_states[
            static_cast<std::size_t>(observation) * observation_stride
        ] = state;
    }
    simulate_one_step(
        model,
        transitions[observation_count - 1U],
        uniforms,
        normal_cache,
        state
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
    philox::NormalPairCache normal_cache;
    simulate_one_step(
        model, initial_stub_transition, uniforms, normal_cache, state
    );
    if (observation_count == 1U) return state;
    observed_states[0U] = state;
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, regular_transition, uniforms, normal_cache, state
        );
        observed_states[
            static_cast<std::size_t>(observation) * observation_stride
        ] = state;
    }
    simulate_one_step(
        model, regular_transition, uniforms, normal_cache, state
    );
    return state;
}

}  // namespace ai_factory::workbench::model::cir
