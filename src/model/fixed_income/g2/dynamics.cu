// Exact G2 state and optional state-integral simulation.
#pragma once

#include "model/fixed_income/g2/dynamics.cuh"

// Reuse stable moments of each Gaussian mean-reverting factor.
#include "model/fixed_income/common/mean_reverting_gaussian.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::g2 {

// ======================== Model-specific dynamics =========================
namespace {

// Return the exact covariance of the two filtered state innovations.
__device__ __forceinline__ float state_covariance(
    const ProcessParameters& parameters,
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
    const ProcessParameters& parameters,
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
__device__ __forceinline__ IntegralMoments integral_moments(
    const ProcessParameters& parameters,
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

__device__ __forceinline__ PreparedModel prepare_model(
    const ProcessParameters& parameters
) {
    return {parameters};
}

// Prepare the Cholesky coefficients of both correlated state innovations.
__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    const ProcessParameters& parameters = model.process;
    const float variance_x = mean_reverting_gaussian::state_variance(
        parameters.mean_reversion_x, parameters.volatility_x, delta_t
    );
    const float variance_y = mean_reverting_gaussian::state_variance(
        parameters.mean_reversion_y, parameters.volatility_y, delta_t
    );
    const float standard_deviation_x = sqrtf(fmaxf(variance_x, 0.0f));
    const float y_x_loading = standard_deviation_x > 0.0f
        ? state_covariance(parameters, delta_t) / standard_deviation_x
        : 0.0f;
    return {
        expf(-parameters.mean_reversion_x * delta_t),
        standard_deviation_x,
        expf(-parameters.mean_reversion_y * delta_t),
        y_x_loading,
        sqrtf(fmaxf(variance_y - y_x_loading * y_x_loading, 0.0f)),
    };
}

__device__ __forceinline__ void prepare_calendar(
    const PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
) {
    for (std::uint32_t interval = 0U; interval < interval_count; ++interval) {
        transitions[interval] = g2::prepare_transition(
            model, static_cast<float>(interval_steps[interval]) * delta_t
        );
    }
}

// Apply one exact correlated transition to both factor states.
__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float state_x_normal,
    float state_y_normal,
    State& state
) {
    state.state_x = fmaf(
        transition.decay_x,
        state.state_x,
        transition.state_x_standard_deviation * state_x_normal
    );
    const float y_noise = fmaf(
        transition.state_y_x_normal_loading,
        state_x_normal,
        transition.state_y_independent_standard_deviation * state_y_normal
    );
    state.state_y = fmaf(transition.decay_y, state.state_y, y_noise);
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& model
) {
    return model.initial_state;
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel&,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) {
    const float state_x_normal = philox::next_normal(uniforms, normals);
    const float state_y_normal = philox::next_normal(uniforms, normals);
    one_step_transition(
        transition, state_x_normal, state_y_normal, state
    );
}

}  // namespace

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    State initial_state,
    philox::PhiloxKey key,
    std::size_t path
) {
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    simulate_one_step(model, transition, uniforms, normals, initial_state);
    return initial_state;
}

__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y
) {
    State state = initial_state;
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
        observed_states_x[output] = state.state_x;
        observed_states_y[output] = state.state_y;
    }
    simulate_one_step(
        model, transitions[observation_count - 1U], uniforms, normals, state
    );
    return state;
}

__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y
) {
    State state = initial_state;
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normals;
    simulate_one_step(
        model, initial_stub_transition, uniforms, normals, state
    );
    if (observation_count == 1U) return state;
    observed_states_x[0U] = state.state_x;
    observed_states_y[0U] = state.state_y;

    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(model, regular_transition, uniforms, normals, state);
        const std::size_t output =
            static_cast<std::size_t>(observation) * observation_stride;
        observed_states_x[output] = state.state_x;
        observed_states_y[output] = state.state_y;
    }
    simulate_one_step(model, regular_transition, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters, float delta_t
) {
    const PreparedModel model = DynamicsPolicy::prepare_model(parameters);
    return {model, DynamicsPolicy::prepare_transition(model, delta_t)};
}

__device__ __forceinline__ DynamicsPolicy::PreparedModel
DynamicsPolicy::prepare_model(const Parameters& parameters) {
    PreparedModel model = g2::prepare_model(parameters.process);
    model.initial_state = parameters.initial_state;
    return model;
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& model, float delta_t
) {
    return g2::prepare_transition(model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return g2::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& model) {
    return g2::initial_state(model);
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
    const PreparedModel& model,
    const PreparedTransition& transition,
    RandomContext& random,
    State& state
) {
    g2::simulate_one_step(
        model, transition, random.uniforms, random.normals, state
    );
}

namespace joint {

// ========================= Common joint dynamics ===========================

// Prepare a three-dimensional Cholesky transition for X, Y, and their integral.
__device__ __forceinline__ PreparedTransition prepare_transition(
    const g2::PreparedModel& model,
    float delta_t
) {
    const ProcessParameters& parameters = model.process;
    const g2::PreparedTransition state_transition =
        g2::prepare_transition(model, delta_t);
    const IntegralMoments moments =
        g2::integral_moments(parameters, delta_t);
    const float covariance_x_integral =
        parameters.volatility_x * parameters.volatility_x
            * moments.state_x_loading * moments.state_x_loading * 0.5f
        + parameters.correlation
            * parameters.volatility_x * parameters.volatility_y
            * state_cross_integral_kernel(
                parameters.mean_reversion_x,
                parameters.mean_reversion_y,
                delta_t
            );
    const float covariance_y_integral =
        parameters.volatility_y * parameters.volatility_y
            * moments.state_y_loading * moments.state_y_loading * 0.5f
        + parameters.correlation
            * parameters.volatility_x * parameters.volatility_y
            * state_cross_integral_kernel(
                parameters.mean_reversion_y,
                parameters.mean_reversion_x,
                delta_t
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

__device__ __forceinline__ void prepare_calendar(
    const g2::PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
) {
    for (std::uint32_t interval = 0U; interval < interval_count; ++interval) {
        transitions[interval] = g2::joint::prepare_transition(
            model, static_cast<float>(interval_steps[interval]) * delta_t
        );
    }
}

// Apply the exact joint state and integral transition.
__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float state_x_normal,
    float state_y_normal,
    float integral_normal,
    State& joint_state
) {
    const g2::State previous_state = joint_state.state;
    joint_state.state.state_x = fmaf(
        transition.decay_x,
        previous_state.state_x,
        transition.state_x_standard_deviation * state_x_normal
    );
    const float state_y_noise = fmaf(
        transition.state_y_x_normal_loading,
        state_x_normal,
        transition.state_y_independent_standard_deviation * state_y_normal
    );
    joint_state.state.state_y = fmaf(
        transition.decay_y, previous_state.state_y, state_y_noise
    );
    float integral_noise = fmaf(
        transition.integral_x_normal_loading,
        state_x_normal,
        transition.integral_independent_standard_deviation * integral_normal
    );
    integral_noise = fmaf(
        transition.integral_y_normal_loading, state_y_normal, integral_noise
    );
    joint_state.state_integral = fmaf(
        transition.integral_state_x_loading,
        previous_state.state_x,
        joint_state.state_integral + integral_noise
    );
    joint_state.state_integral = fmaf(
        transition.integral_state_y_loading,
        previous_state.state_y,
        joint_state.state_integral
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const g2::PreparedModel&,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) {
    const float state_x_normal = philox::next_normal(uniforms, normals);
    const float state_y_normal = philox::next_normal(uniforms, normals);
    const float integral_normal = philox::next_normal(uniforms, normals);
    one_step_transition(
        transition,
        state_x_normal,
        state_y_normal,
        integral_normal,
        state
    );
}

}  // namespace

__device__ __forceinline__ State simulate_terminal_state(
    const g2::PreparedModel& model,
    const PreparedTransition& transition,
    g2::State initial_state,
    philox::PhiloxKey key,
    std::size_t path
) {
    State joint_state{initial_state, 0.0f};
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normals;
    simulate_one_step(model, transition, uniforms, normals, joint_state);
    return joint_state;
}

__device__ __forceinline__ State simulate_on_calendar(
    const g2::PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    g2::State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y,
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
        observed_states_x[output] = state.state.state_x;
        observed_states_y[output] = state.state.state_y;
        observed_integrated_states[output] = state.state_integral;
    }
    simulate_one_step(
        model, transitions[observation_count - 1U], uniforms, normals, state
    );
    return state;
}

__device__ __forceinline__ State simulate_on_regular_grid(
    const g2::PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    g2::State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y,
    float* __restrict__ observed_integrated_states
) {
    State joint_state{initial_state, 0.0f};
    if (observation_count == 0U) return joint_state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normals;
    simulate_one_step(
        model, initial_stub_transition, uniforms, normals, joint_state
    );
    if (observation_count == 1U) return joint_state;
    observed_states_x[0U] = joint_state.state.state_x;
    observed_states_y[0U] = joint_state.state.state_y;
    observed_integrated_states[0U] = joint_state.state_integral;

    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, regular_transition, uniforms, normals, joint_state
        );
        const std::size_t output =
            static_cast<std::size_t>(observation) * observation_stride;
        observed_states_x[output] = joint_state.state.state_x;
        observed_states_y[output] = joint_state.state.state_y;
        observed_integrated_states[output] = joint_state.state_integral;
    }
    simulate_one_step(
        model, regular_transition, uniforms, normals, joint_state
    );
    return joint_state;
}

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters, float delta_t
) {
    const PreparedModel model = DynamicsPolicy::prepare_model(parameters);
    return {model, DynamicsPolicy::prepare_transition(model, delta_t)};
}

__device__ __forceinline__ DynamicsPolicy::PreparedModel
DynamicsPolicy::prepare_model(const Parameters& parameters) {
    return g2::DynamicsPolicy::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& model, float delta_t
) {
    return joint::prepare_transition(model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return DynamicsPolicy::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& model) {
    return {g2::initial_state(model), 0.0f};
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
    const PreparedModel& model,
    const PreparedTransition& transition,
    RandomContext& random,
    State& state
) {
    joint::simulate_one_step(
        model, transition, random.uniforms, random.normals, state
    );
}

}  // namespace joint
}  // namespace ai_factory::workbench::model::g2
