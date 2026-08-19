// Exact Vasicek state and optional state-integral simulation.
#pragma once

#include "model/fixed_income/vasicek/dynamics.cuh"
#include "model/fixed_income/common/mean_reverting_gaussian.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::vasicek {

// ======================== Model-specific dynamics =========================

// Evaluate the exact loading of the current state in its future integral.
__device__ __forceinline__ float integral_state_loading(
    float mean_reversion,
    float delta
) {
    return mean_reverting_gaussian::integral_state_loading(
        mean_reversion, delta
    );
}

// Evaluate the integral variance stably near zero mean reversion.
__device__ __forceinline__ float integral_variance(
    const VasicekProcessParameters& parameters,
    float delta
) {
    return mean_reverting_gaussian::integral_variance(
        parameters.mean_reversion, parameters.volatility, delta
    );
}

// Share one decay evaluation between the loading and integral variance.
__device__ __forceinline__ VasicekIntegralMoments integral_moments(
    const VasicekProcessParameters& parameters,
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

// ======================= Common state-only dynamics ========================

// Prepare one exact Gaussian transition for the Vasicek state alone.
__device__ __forceinline__ VasicekExactTransition prepare_model(
    const VasicekProcessParameters& parameters,
    float time_interval
) {
    const float a = parameters.mean_reversion;
    const float volatility_squared =
        parameters.volatility * parameters.volatility;
    const float one_minus_decay = -expm1f(-a * time_interval);
    const float decay = 1.0f - one_minus_decay;
    const float state_variance =
        mean_reverting_gaussian::state_variance_from_decay(
            a, volatility_squared, decay, one_minus_decay
        );
    return {
        decay,
        parameters.long_term_mean * one_minus_decay,
        sqrtf(state_variance),
    };
}

// Simulate one exact Gaussian Vasicek-state transition.
__device__ __forceinline__ void one_step_transition(
    const VasicekExactTransition& model,
    float state_normal,
    float& state
) {
    const float noise = model.state_standard_deviation * state_normal;
    state = fmaf(model.decay, state, model.mean_increment + noise);
}

// Apply one exact transition from time zero to the requested maturity.
__device__ __forceinline__ float simulate_terminal_state(
    const VasicekExactTransition& model,
    float initial_state,
    float state_normal
) {
    one_step_transition(model, state_normal, initial_state);
    return initial_state;
}

// Write simple states in a date-major grid and return the terminal one.
__device__ __forceinline__ float simulate_on_regular_grid(
    const VasicekExactTransition& initial_stub_model,
    const VasicekExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states
) {
    float state = initial_state;
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    // Reach the first observation through its possibly shorter stub.
    const float first_state_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(initial_stub_model, first_state_normal, state);
    if (observation_count == 1U) return state;
    std::size_t output_index = path;
    observed_states[output_index] = state;

    // Store only pre-terminal states with one running date-major offset.
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        const float state_normal =
            philox::next_normal(uniforms, normal_cache);
        one_step_transition(regular_model, state_normal, state);
        output_index += path_count;
        observed_states[output_index] = state;
    }

    // The last transition is returned without a global-memory write.
    const float terminal_state_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(regular_model, terminal_state_normal, state);
    return state;
}

namespace joint {

// ========================= Common joint dynamics ===========================

// Prepare one exact joint Gaussian transition for the state and its integral.
__device__ __forceinline__ VasicekJointExactTransition prepare_model(
    const VasicekProcessParameters& parameters,
    float time_interval
) {
    const float a = parameters.mean_reversion;
    const float sigma2 = parameters.volatility * parameters.volatility;
    const float one_minus_decay = -expm1f(-a * time_interval);
    const float decay = 1.0f - one_minus_decay;
    const float state_variance =
        mean_reverting_gaussian::state_variance_from_decay(
            a, sigma2, decay, one_minus_decay
        );
    const float state_standard_deviation = sqrtf(state_variance);
    const float covariance =
        mean_reverting_gaussian::state_integral_covariance_from_decay(
            a, sigma2, one_minus_decay
        );
    const float integral_state_loading = one_minus_decay / a;
    const float integral_state_normal_loading =
        state_standard_deviation > 0.0f
        ? covariance / state_standard_deviation
        : 0.0f;
    const float independent_variance = fmaxf(
        mean_reverting_gaussian::integral_variance_from_decay(
            a,
            sigma2,
            time_interval,
            decay,
            one_minus_decay
        )
            - integral_state_normal_loading
                * integral_state_normal_loading,
        0.0f
    );

    return {
        decay,
        parameters.long_term_mean * one_minus_decay,
        state_standard_deviation,
        integral_state_loading,
        parameters.long_term_mean
            * (time_interval - integral_state_loading),
        integral_state_normal_loading,
        sqrtf(independent_variance),
    };
}

// Simulate the exact correlated state and state-integral transition.
__device__ __forceinline__ void one_step_transition(
    const VasicekJointExactTransition& model,
    float state_normal,
    float integral_normal,
    VasicekJointState& joint_state
) {
    const float previous_state = joint_state.state;
    const float state_noise =
        model.state_standard_deviation * state_normal;
    const float integral_noise = fmaf(
        model.integral_state_normal_loading,
        state_normal,
        model.integral_independent_standard_deviation * integral_normal
    );

    joint_state.state = fmaf(
        model.decay,
        previous_state,
        model.state_mean_increment + state_noise
    );
    joint_state.state_integral = fmaf(
        model.integral_state_loading,
        previous_state,
        joint_state.state_integral
            + model.integral_mean_increment
            + integral_noise
    );
}

// Apply one exact joint transition from time zero to maturity.
__device__ __forceinline__ VasicekJointState simulate_terminal_state(
    const VasicekJointExactTransition& model,
    float initial_state,
    float state_normal,
    float integral_normal
) {
    VasicekJointState joint_state{initial_state, 0.0f};
    one_step_transition(model, state_normal, integral_normal, joint_state);
    return joint_state;
}

// Write joint states in a date-major grid and return the terminal one.
__device__ __forceinline__ VasicekJointState simulate_on_regular_grid(
    const VasicekJointExactTransition& initial_stub_model,
    const VasicekJointExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
) {
    VasicekJointState joint_state{initial_state, 0.0f};
    if (observation_count == 0U) return joint_state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    // Reach the first observation through its possibly shorter stub.
    const float first_state_normal =
        philox::next_normal(uniforms, normal_cache);
    const float first_integral_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        initial_stub_model,
        first_state_normal,
        first_integral_normal,
        joint_state
    );
    if (observation_count == 1U) return joint_state;
    std::size_t output_index = path;
    observed_states[output_index] = joint_state.state;
    observed_integrated_states[output_index] = joint_state.state_integral;

    // Store only pre-terminal states with one running date-major offset.
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        const float state_normal = philox::next_normal(uniforms, normal_cache);
        const float integral_normal =
            philox::next_normal(uniforms, normal_cache);
        one_step_transition(
            regular_model, state_normal, integral_normal, joint_state
        );
        output_index += path_count;
        observed_states[output_index] = joint_state.state;
        observed_integrated_states[output_index] =
            joint_state.state_integral;
    }

    // Consume the final pair without writing the maturity state globally.
    const float terminal_state_normal =
        philox::next_normal(uniforms, normal_cache);
    const float terminal_integral_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        regular_model,
        terminal_state_normal,
        terminal_integral_normal,
        joint_state
    );
    return joint_state;
}

}  // namespace joint
}  // namespace ai_factory::workbench::model::vasicek
