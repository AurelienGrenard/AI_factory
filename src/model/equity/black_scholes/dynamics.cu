// Exact Black-Scholes preparation and path simulation.
#include "model/equity/black_scholes/dynamics.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::black_scholes {

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float variance = parameters.volatility * parameters.volatility;
    return {
        logf(parameters.spot),
        parameters.risk_free_rate - parameters.dividend_yield
            - 0.5f * variance,
        parameters.volatility,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    return {
        model.drift_rate * delta_t,
        model.volatility * sqrtf(delta_t),
    };
}

__device__ __forceinline__ void prepare_calendar(
    const PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
) {
    for (std::uint32_t interval = 0U;
         interval < interval_count;
         ++interval) {
        transitions[interval] = prepare_transition(
            model,
            static_cast<float>(interval_steps[interval]) * delta_t
        );
    }
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& model
) {
    return {model.initial_log_spot};
}

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float brownian_normal,
    State& state
) {
    state.log_spot = fmaf(
        transition.standard_deviation,
        brownian_normal,
        state.log_spot + transition.drift
    );
}

// ==================== Model-specific implementation =======================

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel&,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) {
    one_step_transition(
        transition, philox::next_normal(uniforms, normals), state
    );
}

}  // namespace

// ======================== Common equity dynamics =========================

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path
) {
    State state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(model, transition, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ MeanPathResult simulate_mean_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
) {
    State state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = expf(state.log_spot);
    for (std::uint32_t interval = 0U;
         interval < interval_count;
         ++interval) {
        simulate_one_step(model, transition, uniforms, normals, state);
        sum += expf(state.log_spot);
    }
    const double observation_count =
        static_cast<double>(interval_count) + 1.0;
    return {static_cast<float>(sum / observation_count)};
}

__device__ __forceinline__ GeometricMeanPathResult
simulate_geometric_mean_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
) {
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    const float terminal_normal = philox::next_normal(uniforms, normals);
    const float residual_normal = philox::next_normal(uniforms, normals);
    const float count = static_cast<float>(interval_count);
    const float terminal_standard_deviation =
        transition.standard_deviation * sqrtf(count);
    const float shared_coefficient = 0.5f * terminal_standard_deviation;
    const float residual_variance = transition.standard_deviation
        * transition.standard_deviation * count * (count - 1.0f)
        / (12.0f * (count + 1.0f));
    const float average_log_spot = model.initial_log_spot
        + 0.5f * count * transition.drift
        + shared_coefficient * terminal_normal
        + sqrtf(residual_variance) * residual_normal;
    return {expf(average_log_spot)};
}

__device__ __forceinline__ MaximumPathResult simulate_maximum_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
) {
    State state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = expf(state.log_spot);
    for (std::uint32_t interval = 0U;
         interval < interval_count;
         ++interval) {
        simulate_one_step(model, transition, uniforms, normals, state);
        maximum = fmaxf(maximum, expf(state.log_spot));
    }
    return {maximum};
}

__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
) {
    State state = initial_state(model);
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    for (std::uint32_t observation = 0U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, transitions[observation], uniforms, normals, state
        );
        observed_spots[
            static_cast<std::size_t>(observation) * observation_stride
        ] = expf(state.log_spot);
    }
    simulate_one_step(
        model,
        transitions[observation_count - 1U],
        uniforms,
        normals,
        state
    );
    return state;
}

__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
) {
    State state = initial_state(model);
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(
        model, initial_stub_transition, uniforms, normals, state
    );
    if (observation_count == 1U) return state;
    observed_spots[0U] = expf(state.log_spot);
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, regular_transition, uniforms, normals, state
        );
        observed_spots[
            static_cast<std::size_t>(observation) * observation_stride
        ] = expf(state.log_spot);
    }
    simulate_one_step(
        model, regular_transition, uniforms, normals, state
    );
    return state;
}

}  // namespace ai_factory::workbench::black_scholes
