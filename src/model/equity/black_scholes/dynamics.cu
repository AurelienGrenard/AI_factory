// Exact Black-Scholes preparation and path simulation.
#include "model/equity/black_scholes/dynamics.cuh"

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::black_scholes {

__device__ __forceinline__ BlackScholesPreparedParameters prepare_model(
    const BlackScholesModelParameters& parameters,
    float time_interval
) {
    const float variance = parameters.volatility * parameters.volatility;
    return {
        logf(parameters.spot),
        (parameters.risk_free_rate - parameters.dividend_yield
            - 0.5f * variance) * time_interval,
        parameters.volatility * sqrtf(time_interval),
    };
}

__device__ __forceinline__ BlackScholesPreparedParameters prepare_model(
    const BlackScholesModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    return prepare_model(
        parameters, maturity / static_cast<float>(num_steps)
    );
}

__device__ __forceinline__ BlackScholesState initial_state(
    const BlackScholesPreparedParameters& model
) {
    return {model.initial_log_spot};
}

__device__ __forceinline__ void one_step_transition(
    const BlackScholesPreparedParameters& model,
    float brownian_normal,
    BlackScholesState& state
) {
    state.log_spot = fmaf(
        model.standard_deviation,
        brownian_normal,
        state.log_spot + model.drift
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const BlackScholesPreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    BlackScholesState& state
) {
    one_step_transition(
        model, philox::next_normal(uniforms, normal_cache), state
    );
}

}  // namespace

__device__ __forceinline__ BlackScholesState simulate_terminal_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
) {
    BlackScholesState state = initial_state(model);
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normal_cache;
    simulate_one_step(model, uniforms, normal_cache, state);
    return state;
}

__device__ __forceinline__ BlackScholesMeanPathResult simulate_mean_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    BlackScholesState state = initial_state(model);
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normal_cache;
    double spot_sum = static_cast<double>(expf(state.log_spot));
    for (std::size_t step_index = 0U; step_index < num_steps; ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        spot_sum += static_cast<double>(expf(state.log_spot));
    }
    return {
        state,
        static_cast<float>(spot_sum / (static_cast<double>(num_steps) + 1.0)),
    };
}

// Simulate the joint Gaussian law of the terminal log-price and the discrete
// average log-price directly; no intermediate grid points are constructed.
__device__ __forceinline__ BlackScholesGeometricMeanPathResult
simulate_geometric_mean_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normal_cache;
    const float terminal_normal = philox::next_normal(uniforms, normal_cache);
    const float residual_normal = philox::next_normal(uniforms, normal_cache);
    const float step_count = static_cast<float>(num_steps);
    const float terminal_standard_deviation =
        model.standard_deviation * sqrtf(step_count);
    const float shared_coefficient = 0.5f * terminal_standard_deviation;
    // This positive form avoids cancelling average_variance against the
    // shared terminal component; it is exactly zero for a one-step grid.
    const float residual_variance = model.standard_deviation
        * model.standard_deviation * step_count * (step_count - 1.0f)
        / (12.0f * (step_count + 1.0f));
    const float residual_standard_deviation = sqrtf(residual_variance);
    const float terminal_log_spot = fmaf(
        terminal_standard_deviation,
        terminal_normal,
        model.initial_log_spot + step_count * model.drift
    );
    const float average_log_spot = model.initial_log_spot
        + 0.5f * step_count * model.drift
        + shared_coefficient * terminal_normal
        + residual_standard_deviation * residual_normal;
    return {{terminal_log_spot}, expf(average_log_spot)};
}

__device__ __forceinline__ BlackScholesTwoTimePathResult simulate_at_two_times(
    const BlackScholesPreparedParameters& first_model,
    const BlackScholesPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
) {
    BlackScholesState state = initial_state(first_model);
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normal_cache;
    simulate_one_step(first_model, uniforms, normal_cache, state);
    const BlackScholesState first_state = state;
    simulate_one_step(second_model, uniforms, normal_cache, state);
    return {first_state, state};
}

__device__ __forceinline__ BlackScholesMaximumPathResult simulate_maximum_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    BlackScholesState state = initial_state(model);
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normal_cache;
    float maximum_spot = expf(state.log_spot);
    for (std::size_t step_index = 0U; step_index < num_steps; ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        maximum_spot = fmaxf(maximum_spot, expf(state.log_spot));
    }
    return {state, maximum_spot};
}

__device__ __forceinline__ BlackScholesState simulate_on_regular_grid(
    const BlackScholesPreparedParameters& initial_stub_model,
    const BlackScholesPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
) {
    BlackScholesState state = initial_state(initial_stub_model);
    philox::UniformSequence uniforms(key, static_cast<std::uint64_t>(path));
    philox::NormalPairCache normal_cache;
    simulate_one_step(initial_stub_model, uniforms, normal_cache, state);
    if (exercise_count == 1U) return state;
    std::size_t output_index = path;
    observed_spots[output_index] = expf(state.log_spot);
    for (std::uint32_t exercise = 1U;
         exercise + 1U < exercise_count;
         ++exercise) {
        simulate_one_step(regular_model, uniforms, normal_cache, state);
        output_index += path_count;
        observed_spots[output_index] = expf(state.log_spot);
    }
    simulate_one_step(regular_model, uniforms, normal_cache, state);
    return state;
}

}  // namespace ai_factory::workbench::black_scholes
