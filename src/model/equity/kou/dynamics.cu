#include "model/equity/kou/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::kou {

__device__ __forceinline__ KouPreparedParameters prepare_model(
    const KouModelParameters& parameters,
    float time_interval
) {
    const float jump_martingale =
        parameters.up_probability * parameters.positive_jump_rate
            / (parameters.positive_jump_rate - 1.0f)
        + (1.0f - parameters.up_probability) * parameters.negative_jump_rate
            / (parameters.negative_jump_rate + 1.0f)
        - 1.0f;
    const float diffusion_variance =
        parameters.volatility * parameters.volatility;
    const float poisson_mean = parameters.jump_intensity * time_interval;

    return {
        logf(parameters.spot),
        (
            parameters.risk_free_rate
            - parameters.dividend_yield
            - parameters.jump_intensity * jump_martingale
            - 0.5f * diffusion_variance
        ) * time_interval,
        parameters.volatility * sqrtf(time_interval),
        poisson_mean,
        expf(-poisson_mean),
        parameters.up_probability,
        1.0f / parameters.positive_jump_rate,
        1.0f / parameters.negative_jump_rate,
    };
}

__device__ __forceinline__ KouPreparedParameters prepare_model(
    const KouModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    return prepare_model(
        parameters,
        maturity / static_cast<float>(num_steps)
    );
}

__device__ __forceinline__ KouState initial_state(
    const KouPreparedParameters& model
) {
    return {model.initial_log_spot};
}

__device__ __forceinline__ void one_step_transition(
    const KouPreparedParameters& model,
    float diffusion_normal,
    float jump_log_sum,
    KouState& state
) {
    state.log_spot +=
        model.drift_dt
        + model.diffusion_std * diffusion_normal
        + jump_log_sum;
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const KouPreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    KouState& state
) {
    const std::uint32_t jump_count = philox::poisson_from_uniform(
        uniforms.next(),
        model.poisson_mean,
        model.zero_jump_probability
    );

    // A path-local UniformSequence makes the variable number of jump draws
    // collision-free without reserving a fixed Philox range per time step.
    float jump_log_sum = 0.0f;
    for (std::uint32_t jump_index = 0U;
         jump_index < jump_count;
         ++jump_index) {
        const bool upward = uniforms.next() < model.up_probability;
        const float magnitude = -logf(uniforms.next());
        jump_log_sum += upward
            ? magnitude * model.inverse_positive_jump_rate
            : -magnitude * model.inverse_negative_jump_rate;
    }

    one_step_transition(
        model,
        philox::next_normal(uniforms, normals),
        jump_log_sum,
        state
    );
}

}  // namespace

__device__ __forceinline__ KouState simulate_terminal_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
) {
    KouState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(model, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ KouMeanPathResult simulate_mean_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    KouState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = expf(state.log_spot);
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        sum += expf(state.log_spot);
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {static_cast<float>(sum / observation_count)};
}

__device__ __forceinline__ KouGeometricMeanPathResult
simulate_geometric_mean_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    KouState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double log_sum = state.log_spot;
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        log_sum += state.log_spot;
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {
        expf(static_cast<float>(log_sum / observation_count)),
    };
}

__device__ __forceinline__ KouTwoTimePathResult simulate_at_two_times(
    const KouPreparedParameters& first_model,
    const KouPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
) {
    KouState state = initial_state(first_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(first_model, uniforms, normals, state);
    const float first_spot = expf(state.log_spot);
    simulate_one_step(second_model, uniforms, normals, state);
    return {first_spot, expf(state.log_spot)};
}

__device__ __forceinline__ KouMaximumPathResult simulate_maximum_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    KouState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = expf(state.log_spot);
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        maximum = fmaxf(maximum, expf(state.log_spot));
    }
    return {maximum};
}

__device__ __forceinline__ KouState simulate_on_regular_grid(
    const KouPreparedParameters& initial_stub_model,
    const KouPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* observed_spots
) {
    KouState state = initial_state(initial_stub_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(initial_stub_model, uniforms, normals, state);
    if (exercise_count == 1U) {
        return state;
    }

    std::size_t output_index = path;
    observed_spots[output_index] = expf(state.log_spot);
    for (std::uint32_t exercise_index = 1U;
         exercise_index + 1U < exercise_count;
         ++exercise_index) {
        simulate_one_step(regular_model, uniforms, normals, state);
        output_index += path_count;
        observed_spots[output_index] = expf(state.log_spot);
    }
    simulate_one_step(regular_model, uniforms, normals, state);
    return state;
}

}  // namespace ai_factory::workbench::kou
