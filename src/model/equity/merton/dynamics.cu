#include "model/equity/merton/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::merton {

__device__ __forceinline__ MertonPreparedParameters prepare_model(
    const MertonModelParameters& parameters,
    float time_interval
) {
    const float jump_variance =
        parameters.jump_log_volatility * parameters.jump_log_volatility;
    const float jump_martingale = expf(
        parameters.jump_log_mean + 0.5f * jump_variance
    ) - 1.0f;
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
        parameters.jump_log_mean,
        parameters.jump_log_volatility,
    };
}

__device__ __forceinline__ MertonPreparedParameters prepare_model(
    const MertonModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    return prepare_model(
        parameters,
        maturity / static_cast<float>(num_steps)
    );
}

__device__ __forceinline__ MertonState initial_state(
    const MertonPreparedParameters& model
) {
    return {model.initial_log_spot};
}

// Conditional on N jumps, their Gaussian log-sizes sum exactly to one normal
// variate with mean N mu_J and standard deviation sqrt(N) sigma_J.
__device__ __forceinline__ void one_step_transition(
    const MertonPreparedParameters& model,
    std::uint32_t jump_count,
    float diffusion_normal,
    float jump_normal,
    MertonState& state
) {
    const float count = static_cast<float>(jump_count);
    const float jump_log_sum =
        count * model.jump_log_mean
        + model.jump_log_volatility * sqrtf(count) * jump_normal;
    state.log_spot +=
        model.drift_dt
        + model.diffusion_std * diffusion_normal
        + jump_log_sum;
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const MertonPreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    MertonState& state
) {
    const std::uint32_t jump_count = philox::poisson_from_uniform(
        uniforms.next(),
        model.poisson_mean,
        model.zero_jump_probability
    );
    const float diffusion_normal = philox::next_normal(uniforms, normals);
    const float jump_normal = jump_count == 0U
        ? 0.0f
        : philox::next_normal(uniforms, normals);
    one_step_transition(
        model,
        jump_count,
        diffusion_normal,
        jump_normal,
        state
    );
}

}  // namespace

__device__ __forceinline__ MertonState simulate_terminal_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
) {
    MertonState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(model, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ MertonMeanPathResult simulate_mean_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    MertonState state = initial_state(model);
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

__device__ __forceinline__ MertonGeometricMeanPathResult
simulate_geometric_mean_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    MertonState state = initial_state(model);
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

__device__ __forceinline__ MertonTwoTimePathResult simulate_at_two_times(
    const MertonPreparedParameters& first_model,
    const MertonPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
) {
    MertonState state = initial_state(first_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(first_model, uniforms, normals, state);
    const float first_spot = expf(state.log_spot);
    simulate_one_step(second_model, uniforms, normals, state);
    return {first_spot, expf(state.log_spot)};
}

__device__ __forceinline__ MertonMaximumPathResult simulate_maximum_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    MertonState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = expf(state.log_spot);
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        maximum = fmaxf(maximum, expf(state.log_spot));
    }
    return {maximum};
}

__device__ __forceinline__ MertonState simulate_on_regular_grid(
    const MertonPreparedParameters& initial_stub_model,
    const MertonPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* observed_spots
) {
    MertonState state = initial_state(initial_stub_model);
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

}  // namespace ai_factory::workbench::merton
