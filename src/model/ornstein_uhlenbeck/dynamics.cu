// Exact Ornstein-Uhlenbeck state and optional state-integral simulation.
#pragma once

#include "model/ornstein_uhlenbeck/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {
namespace {

// Evaluate B(delta) by series when mean_reversion * delta is small.
__device__ __forceinline__ float small_time_integral_state_loading(
    float delta,
    float scaled_time
) {
    float normalized = fmaf(scaled_time, 1.0f / 120.0f, -1.0f / 24.0f);
    normalized = fmaf(scaled_time, normalized, 1.0f / 6.0f);
    normalized = fmaf(scaled_time, normalized, -0.5f);
    normalized = fmaf(scaled_time, normalized, 1.0f);
    return delta * normalized;
}

// Evaluate the small-time integral variance without transcendental functions.
__device__ __forceinline__ float small_time_integral_variance(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float delta,
    float scaled_time
) {
    const float scaled_time2 = scaled_time * scaled_time;
    const float normalized =
        1.0f / 3.0f
        - scaled_time / 4.0f
        + 7.0f * scaled_time2 / 60.0f
        - scaled_time2 * scaled_time / 24.0f;
    return parameters.volatility * parameters.volatility
        * delta * delta * delta * normalized;
}

// Reuse precomputed decay terms in the exact integral variance.
__device__ __forceinline__ float integral_variance_from_decay(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float delta,
    float decay,
    float one_minus_decay
) {
    const float a = parameters.mean_reversion;
    const float scaled_time = a * delta;
    if (fabsf(scaled_time) < 0.02f) {
        return small_time_integral_variance(
            parameters, delta, scaled_time
        );
    }

    const float bracket =
        delta
        - 2.0f * one_minus_decay / a
        + one_minus_decay * (1.0f + decay) / (2.0f * a);
    return parameters.volatility * parameters.volatility
        * bracket / (a * a);
}

}  // namespace

// Evaluate the exact loading of the current state in its future integral.
__device__ __forceinline__ float integral_state_loading(
    float mean_reversion,
    float delta
) {
    const float scaled_time = mean_reversion * delta;
    if (fabsf(scaled_time) < 0.02f) {
        return small_time_integral_state_loading(delta, scaled_time);
    }
    return -expm1f(-scaled_time) / mean_reversion;
}

// Evaluate the integral variance stably near zero mean reversion.
__device__ __forceinline__ float integral_variance(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float delta
) {
    const float a = parameters.mean_reversion;
    const float scaled_time = a * delta;
    if (fabsf(scaled_time) < 0.02f) {
        return small_time_integral_variance(
            parameters, delta, scaled_time
        );
    }
    const float one_minus_decay = -expm1f(-a * delta);
    const float decay = 1.0f - one_minus_decay;
    return integral_variance_from_decay(
        parameters, delta, decay, one_minus_decay
    );
}

// Share one decay evaluation between the loading and integral variance.
__device__ __forceinline__ OrnsteinUhlenbeckIntegralMoments integral_moments(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float delta
) {
    const float a = parameters.mean_reversion;
    const float scaled_time = a * delta;
    if (fabsf(scaled_time) < 0.02f) {
        return {
            small_time_integral_state_loading(delta, scaled_time),
            small_time_integral_variance(parameters, delta, scaled_time),
        };
    }
    const float one_minus_decay = -expm1f(-scaled_time);
    const float decay = 1.0f - one_minus_decay;
    return {
        one_minus_decay / a,
        integral_variance_from_decay(
            parameters, delta, decay, one_minus_decay
        ),
    };
}

// Prepare one exact Gaussian transition for the OU state alone.
__device__ __forceinline__ OrnsteinUhlenbeckExactTransition prepare_model(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float time_interval
) {
    const float a = parameters.mean_reversion;
    const float one_minus_decay = -expm1f(-a * time_interval);
    const float decay = 1.0f - one_minus_decay;
    const float state_variance =
        parameters.volatility * parameters.volatility
        * one_minus_decay * (1.0f + decay) / (2.0f * a);
    return {
        decay,
        sqrtf(state_variance),
    };
}

// Simulate one exact Gaussian OU-state transition.
__device__ __forceinline__ void one_step_transition(
    const OrnsteinUhlenbeckExactTransition& model,
    float state_normal,
    float& state
) {
    state = fmaf(
        model.decay,
        state,
        model.state_standard_deviation * state_normal
    );
}

// Apply one exact transition from time zero to the requested maturity.
__device__ __forceinline__ float simulate_terminal_state(
    const OrnsteinUhlenbeckExactTransition& model,
    float initial_state,
    float state_normal
) {
    one_step_transition(model, state_normal, initial_state);
    return initial_state;
}

// Write simple states in a date-major grid and return the terminal one.
__device__ __forceinline__ float simulate_on_regular_grid(
    const OrnsteinUhlenbeckExactTransition& initial_stub_model,
    const OrnsteinUhlenbeckExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states
) {
    float state = initial_state;
    if (observation_count == 0U) return state;
    const std::uint64_t groups_per_path =
        (static_cast<std::uint64_t>(observation_count) + 3ULL) >> 2U;
    philox::NormalSequence normals(
        key, static_cast<std::uint64_t>(path) * groups_per_path
    );

    // Reach the first observation through its possibly shorter stub.
    one_step_transition(initial_stub_model, normals.next(), state);
    if (observation_count == 1U) return state;
    std::size_t output_index = path;
    observed_states[output_index] = state;

    // Store only pre-terminal states with one running date-major offset.
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        one_step_transition(regular_model, normals.next(), state);
        output_index += path_count;
        observed_states[output_index] = state;
    }

    // The last transition is returned without a global-memory write.
    one_step_transition(regular_model, normals.next(), state);
    return state;
}

namespace joint {

// Prepare one exact joint Gaussian transition for the state and its integral.
__device__ __forceinline__ OrnsteinUhlenbeckJointExactTransition prepare_model(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float time_interval
) {
    const float a = parameters.mean_reversion;
    const float sigma2 = parameters.volatility * parameters.volatility;
    const float one_minus_decay = -expm1f(-a * time_interval);
    const float decay = 1.0f - one_minus_decay;
    const float state_variance =
        sigma2 * one_minus_decay * (1.0f + decay) / (2.0f * a);
    const float state_standard_deviation = sqrtf(state_variance);
    const float covariance =
        sigma2 * one_minus_decay * one_minus_decay / (2.0f * a * a);
    const float integral_state_normal_loading =
        state_standard_deviation > 0.0f
        ? covariance / state_standard_deviation
        : 0.0f;
    const float independent_variance = fmaxf(
        integral_variance_from_decay(
            parameters, time_interval, decay, one_minus_decay
        )
            - integral_state_normal_loading
                * integral_state_normal_loading,
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

// Simulate the exact correlated state and state-integral transition.
__device__ __forceinline__ void one_step_transition(
    const OrnsteinUhlenbeckJointExactTransition& model,
    float state_normal,
    float integral_normal,
    OrnsteinUhlenbeckJointState& joint_state
) {
    const float previous_state = joint_state.state;
    const float state_noise =
        model.state_standard_deviation * state_normal;
    const float integral_noise = fmaf(
        model.integral_state_normal_loading,
        state_normal,
        model.integral_independent_standard_deviation * integral_normal
    );

    joint_state.state = fmaf(model.decay, previous_state, state_noise);
    joint_state.state_integral = fmaf(
        model.integral_state_loading,
        previous_state,
        joint_state.state_integral + integral_noise
    );
}

// Apply one exact joint transition from time zero to maturity.
__device__ __forceinline__ OrnsteinUhlenbeckJointState simulate_terminal_state(
    const OrnsteinUhlenbeckJointExactTransition& model,
    float initial_state,
    float state_normal,
    float integral_normal
) {
    OrnsteinUhlenbeckJointState joint_state{initial_state, 0.0f};
    one_step_transition(model, state_normal, integral_normal, joint_state);
    return joint_state;
}

// Write joint states in a date-major grid and return the terminal one.
__device__ __forceinline__ OrnsteinUhlenbeckJointState simulate_on_regular_grid(
    const OrnsteinUhlenbeckJointExactTransition& initial_stub_model,
    const OrnsteinUhlenbeckJointExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
) {
    OrnsteinUhlenbeckJointState joint_state{initial_state, 0.0f};
    if (observation_count == 0U) return joint_state;
    const std::uint64_t groups_per_path =
        (static_cast<std::uint64_t>(observation_count) + 1ULL) >> 1U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::NormalSequence normals(key, first_group);

    // Reach the first observation through its possibly shorter stub.
    const float first_state_normal = normals.next();
    const float first_integral_normal = normals.next();
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
        const float state_normal = normals.next();
        const float integral_normal = normals.next();
        one_step_transition(
            regular_model, state_normal, integral_normal, joint_state
        );
        output_index += path_count;
        observed_states[output_index] = joint_state.state;
        observed_integrated_states[output_index] =
            joint_state.state_integral;
    }

    // Consume the final pair without writing the maturity state globally.
    const float terminal_state_normal = normals.next();
    const float terminal_integral_normal = normals.next();
    one_step_transition(
        regular_model,
        terminal_state_normal,
        terminal_integral_normal,
        joint_state
    );
    return joint_state;
}

}  // namespace joint
}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
