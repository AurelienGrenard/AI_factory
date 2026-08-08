// Exact G2 state and optional state-integral simulation.
#pragma once

#include "model/fixed_income/g2/dynamics.cuh"

// Reuse stable moments of each Gaussian mean-reverting factor.
#include "model/fixed_income/common/mean_reverting_gaussian.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::g2 {
namespace {

// Return the exact covariance of the two filtered state innovations.
__device__ __forceinline__ float state_covariance(
    const G2ProcessParameters& parameters,
    float delta
) {
    const float sum =
        parameters.mean_reversion_x + parameters.mean_reversion_y;
    return parameters.correlation
        * parameters.volatility_x * parameters.volatility_y
        * (-expm1f(-sum * delta)) / sum;
}

// Return Cov(integral X, integral Y) with a stable small-time series.
__device__ __forceinline__ float cross_integral_covariance(
    const G2ProcessParameters& parameters,
    float delta
) {
    const float a = parameters.mean_reversion_x;
    const float b = parameters.mean_reversion_y;
    const float scale = fmaxf(a, b) * delta;
    float integral = 0.0f;
    if (fabsf(scale) < 0.02f) {
        const float delta2 = delta * delta;
        integral = delta2 * delta * (
            1.0f / 3.0f
            - (a + b) * delta / 8.0f
            + (2.0f * a * a + 3.0f * a * b + 2.0f * b * b)
                * delta2 / 60.0f
        );
    } else {
        const float loading_a =
            mean_reverting_gaussian::integral_state_loading(a, delta);
        const float loading_b =
            mean_reverting_gaussian::integral_state_loading(b, delta);
        const float loading_sum =
            mean_reverting_gaussian::integral_state_loading(a + b, delta);
        integral = (
            delta - loading_a - loading_b + loading_sum
        ) / (a * b);
    }
    return parameters.correlation
        * parameters.volatility_x * parameters.volatility_y * integral;
}

// Return integral exp(-a u) B_b(u) du with stable small-time arithmetic.
__device__ __forceinline__ float state_cross_integral_kernel(
    float a,
    float b,
    float delta
) {
    const float scale = fmaxf(a, b) * delta;
    if (fabsf(scale) < 0.02f) {
        const float delta2 = delta * delta;
        return delta2 * (
            0.5f
            - (a + 0.5f * b) * delta / 3.0f
            + (0.5f * a * a + 0.5f * a * b + b * b / 6.0f)
                * delta2 / 4.0f
        );
    }
    const float loading_a =
        mean_reverting_gaussian::integral_state_loading(a, delta);
    const float loading_sum =
        mean_reverting_gaussian::integral_state_loading(a + b, delta);
    return (loading_a - loading_sum) / b;
}

}  // namespace

// Combine both OU integral variances and their cross covariance.
__device__ __forceinline__ G2IntegralMoments integral_moments(
    const G2ProcessParameters& parameters,
    float delta
) {
    const auto moments_x =
        mean_reverting_gaussian::integral_moments(
            parameters.mean_reversion_x, parameters.volatility_x, delta
        );
    const auto moments_y =
        mean_reverting_gaussian::integral_moments(
            parameters.mean_reversion_y, parameters.volatility_y, delta
        );
    return {
        moments_x.state_loading,
        moments_y.state_loading,
        moments_x.variance + moments_y.variance
            + 2.0f * cross_integral_covariance(parameters, delta),
    };
}

// Prepare the Cholesky coefficients of both correlated state innovations.
__device__ __forceinline__ G2ExactTransition prepare_model(
    const G2ProcessParameters& parameters,
    float time_interval
) {
    const float variance_x = mean_reverting_gaussian::state_variance(
        parameters.mean_reversion_x, parameters.volatility_x, time_interval
    );
    const float variance_y = mean_reverting_gaussian::state_variance(
        parameters.mean_reversion_y, parameters.volatility_y, time_interval
    );
    const float standard_deviation_x = sqrtf(fmaxf(variance_x, 0.0f));
    const float y_x_loading = standard_deviation_x > 0.0f
        ? state_covariance(parameters, time_interval) / standard_deviation_x
        : 0.0f;
    return {
        expf(-parameters.mean_reversion_x * time_interval),
        standard_deviation_x,
        expf(-parameters.mean_reversion_y * time_interval),
        y_x_loading,
        sqrtf(fmaxf(variance_y - y_x_loading * y_x_loading, 0.0f)),
    };
}

// Apply one exact correlated transition to both factor states.
__device__ __forceinline__ void one_step_transition(
    const G2ExactTransition& model,
    float state_x_normal,
    float state_y_normal,
    G2State& state
) {
    state.state_x = fmaf(
        model.decay_x,
        state.state_x,
        model.state_x_standard_deviation * state_x_normal
    );
    const float y_noise = fmaf(
        model.state_y_x_normal_loading,
        state_x_normal,
        model.state_y_independent_standard_deviation * state_y_normal
    );
    state.state_y = fmaf(model.decay_y, state.state_y, y_noise);
}

// Apply one transition from time zero to the requested maturity.
__device__ __forceinline__ G2State simulate_terminal_state(
    const G2ExactTransition& model,
    G2State initial_state,
    float state_x_normal,
    float state_y_normal
) {
    one_step_transition(
        model, state_x_normal, state_y_normal, initial_state
    );
    return initial_state;
}

// Write both factor states in date-major arrays and return maturity.
__device__ __forceinline__ G2State simulate_on_regular_grid(
    const G2ExactTransition& initial_stub_model,
    const G2ExactTransition& regular_model,
    G2State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y
) {
    G2State state = initial_state;
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    const float first_x_normal = philox::next_normal(uniforms, normal_cache);
    const float first_y_normal = philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        initial_stub_model, first_x_normal, first_y_normal, state
    );
    if (observation_count == 1U) return state;
    std::size_t output_index = path;
    observed_states_x[output_index] = state.state_x;
    observed_states_y[output_index] = state.state_y;

    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        const float state_x_normal =
            philox::next_normal(uniforms, normal_cache);
        const float state_y_normal =
            philox::next_normal(uniforms, normal_cache);
        one_step_transition(
            regular_model, state_x_normal, state_y_normal, state
        );
        output_index += path_count;
        observed_states_x[output_index] = state.state_x;
        observed_states_y[output_index] = state.state_y;
    }
    const float terminal_x_normal = philox::next_normal(uniforms, normal_cache);
    const float terminal_y_normal = philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        regular_model, terminal_x_normal, terminal_y_normal, state
    );
    return state;
}

namespace joint {

// Prepare a three-dimensional Cholesky transition for X, Y, and their integral.
__device__ __forceinline__ G2JointExactTransition prepare_model(
    const G2ProcessParameters& parameters,
    float time_interval
) {
    const G2ExactTransition state_transition =
        model::g2::prepare_model(parameters, time_interval);
    const G2IntegralMoments moments =
        model::g2::integral_moments(parameters, time_interval);
    const float covariance_x_integral =
        parameters.volatility_x * parameters.volatility_x
            * moments.state_x_loading * moments.state_x_loading * 0.5f
        + parameters.correlation
            * parameters.volatility_x * parameters.volatility_y
            * state_cross_integral_kernel(
                parameters.mean_reversion_x,
                parameters.mean_reversion_y,
                time_interval
            );
    const float covariance_y_integral =
        parameters.volatility_y * parameters.volatility_y
            * moments.state_y_loading * moments.state_y_loading * 0.5f
        + parameters.correlation
            * parameters.volatility_x * parameters.volatility_y
            * state_cross_integral_kernel(
                parameters.mean_reversion_y,
                parameters.mean_reversion_x,
                time_interval
            );

    const float l20 = state_transition.state_x_standard_deviation > 0.0f
        ? covariance_x_integral
            / state_transition.state_x_standard_deviation
        : 0.0f;
    const float l21 =
        state_transition.state_y_independent_standard_deviation > 0.0f
        ? (covariance_y_integral
            - l20 * state_transition.state_y_x_normal_loading)
            / state_transition.state_y_independent_standard_deviation
        : 0.0f;
    const float independent_variance = fmaxf(
        moments.variance - l20 * l20 - l21 * l21,
        0.0f
    );
    return {
        state_transition.decay_x,
        state_transition.state_x_standard_deviation,
        state_transition.decay_y,
        state_transition.state_y_x_normal_loading,
        state_transition.state_y_independent_standard_deviation,
        moments.state_x_loading,
        moments.state_y_loading,
        l20,
        l21,
        sqrtf(independent_variance),
    };
}

// Apply the exact joint state and integral transition.
__device__ __forceinline__ void one_step_transition(
    const G2JointExactTransition& model,
    float state_x_normal,
    float state_y_normal,
    float integral_normal,
    G2JointState& joint_state
) {
    const G2State previous_state = joint_state.state;
    joint_state.state.state_x = fmaf(
        model.decay_x,
        previous_state.state_x,
        model.state_x_standard_deviation * state_x_normal
    );
    const float state_y_noise = fmaf(
        model.state_y_x_normal_loading,
        state_x_normal,
        model.state_y_independent_standard_deviation * state_y_normal
    );
    joint_state.state.state_y = fmaf(
        model.decay_y, previous_state.state_y, state_y_noise
    );
    float integral_noise = fmaf(
        model.integral_x_normal_loading,
        state_x_normal,
        model.integral_independent_standard_deviation * integral_normal
    );
    integral_noise = fmaf(
        model.integral_y_normal_loading, state_y_normal, integral_noise
    );
    joint_state.state_integral = fmaf(
        model.integral_state_x_loading,
        previous_state.state_x,
        joint_state.state_integral + integral_noise
    );
    joint_state.state_integral = fmaf(
        model.integral_state_y_loading,
        previous_state.state_y,
        joint_state.state_integral
    );
}

// Apply one exact joint transition from time zero to maturity.
__device__ __forceinline__ G2JointState simulate_terminal_state(
    const G2JointExactTransition& model,
    G2State initial_state,
    float state_x_normal,
    float state_y_normal,
    float integral_normal
) {
    G2JointState joint_state{initial_state, 0.0f};
    one_step_transition(
        model,
        state_x_normal,
        state_y_normal,
        integral_normal,
        joint_state
    );
    return joint_state;
}

// Write both factors and their integral in date-major arrays.
__device__ __forceinline__ G2JointState simulate_on_regular_grid(
    const G2JointExactTransition& initial_stub_model,
    const G2JointExactTransition& regular_model,
    G2State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y,
    float* __restrict__ observed_integrated_states
) {
    G2JointState joint_state{initial_state, 0.0f};
    if (observation_count == 0U) return joint_state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    const float first_x_normal = philox::next_normal(uniforms, normal_cache);
    const float first_y_normal = philox::next_normal(uniforms, normal_cache);
    const float first_integral_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        initial_stub_model,
        first_x_normal,
        first_y_normal,
        first_integral_normal,
        joint_state
    );
    if (observation_count == 1U) return joint_state;
    std::size_t output_index = path;
    observed_states_x[output_index] = joint_state.state.state_x;
    observed_states_y[output_index] = joint_state.state.state_y;
    observed_integrated_states[output_index] = joint_state.state_integral;

    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        const float state_x_normal =
            philox::next_normal(uniforms, normal_cache);
        const float state_y_normal =
            philox::next_normal(uniforms, normal_cache);
        const float integral_normal =
            philox::next_normal(uniforms, normal_cache);
        one_step_transition(
            regular_model,
            state_x_normal,
            state_y_normal,
            integral_normal,
            joint_state
        );
        output_index += path_count;
        observed_states_x[output_index] = joint_state.state.state_x;
        observed_states_y[output_index] = joint_state.state.state_y;
        observed_integrated_states[output_index] = joint_state.state_integral;
    }
    const float terminal_x_normal = philox::next_normal(uniforms, normal_cache);
    const float terminal_y_normal = philox::next_normal(uniforms, normal_cache);
    const float terminal_integral_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        regular_model,
        terminal_x_normal,
        terminal_y_normal,
        terminal_integral_normal,
        joint_state
    );
    return joint_state;
}

}  // namespace joint
}  // namespace ai_factory::workbench::model::g2
