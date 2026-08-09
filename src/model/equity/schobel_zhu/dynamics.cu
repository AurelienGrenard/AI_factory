#include "model/equity/schobel_zhu/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::schobel_zhu {

__device__ __forceinline__ SchobelZhuPreparedParameters prepare_model(
    const SchobelZhuModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    const float dt = maturity / static_cast<float>(num_steps);
    const float sqrt_dt = sqrtf(dt);
    const float exp_mean_reversion_dt = expf(
        -parameters.mean_reversion * dt
    );
    const float endpoint_variance =
        (1.0f - exp_mean_reversion_dt * exp_mean_reversion_dt)
        / (2.0f * parameters.mean_reversion);
    const float endpoint_increment_correlation =
        (1.0f - exp_mean_reversion_dt)
        / (
            parameters.mean_reversion
            * sqrtf(dt * endpoint_variance)
        );
    const float clamped_endpoint_correlation = fminf(
        1.0f,
        fmaxf(-1.0f, endpoint_increment_correlation)
    );

    return {
        logf(parameters.spot),
        parameters.initial_volatility,
        parameters.long_run_volatility,
        exp_mean_reversion_dt,
        parameters.volatility_of_volatility * sqrtf(endpoint_variance),
        clamped_endpoint_correlation,
        sqrtf(
            fmaxf(
                0.0f,
                1.0f
                    - clamped_endpoint_correlation
                        * clamped_endpoint_correlation
            )
        ),
        (parameters.risk_free_rate - parameters.dividend_yield) * dt,
        sqrt_dt,
        parameters.correlation,
        sqrtf(1.0f - parameters.correlation * parameters.correlation),
    };
}

__device__ __forceinline__ SchobelZhuState initial_state(
    const SchobelZhuPreparedParameters& model
) {
    return {model.initial_log_spot, model.initial_volatility};
}

// The OU endpoint is exact. Its Gaussian innovation is coupled to the
// volatility Brownian increment used by the log-spot Euler step, preserving
// the requested instantaneous asset/volatility correlation.
__device__ __forceinline__ void one_step_transition(
    const SchobelZhuPreparedParameters& model,
    float ou_normal,
    float increment_residual_normal,
    float asset_residual_normal,
    SchobelZhuState& state
) {
    const float volatility_brownian_normal = fmaf(
        model.endpoint_increment_residual,
        increment_residual_normal,
        model.endpoint_increment_correlation * ou_normal
    );
    const float asset_normal = fmaf(
        model.correlation_residual,
        asset_residual_normal,
        model.correlation * volatility_brownian_normal
    );
    const float volatility = state.volatility;
    state.log_spot +=
        model.drift_dt
        - 0.5f * volatility * volatility * model.sqrt_dt * model.sqrt_dt
        + volatility * model.sqrt_dt * asset_normal;
    state.volatility =
        model.long_run_volatility
        + (volatility - model.long_run_volatility)
            * model.exp_mean_reversion_dt
        + model.ou_std * ou_normal;
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const SchobelZhuPreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    SchobelZhuState& state
) {
    const float ou_normal = philox::next_normal(uniforms, normals);
    const float increment_residual_normal = philox::next_normal(
        uniforms,
        normals
    );
    const float asset_residual_normal = philox::next_normal(
        uniforms,
        normals
    );
    one_step_transition(
        model,
        ou_normal,
        increment_residual_normal,
        asset_residual_normal,
        state
    );
}

__device__ __forceinline__ void simulate_steps(
    const SchobelZhuPreparedParameters& model,
    std::size_t num_steps,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    SchobelZhuState& state
) {
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
    }
}

}  // namespace

__device__ __forceinline__ SchobelZhuState simulate_terminal_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    SchobelZhuState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(model, num_steps, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ SchobelZhuMeanPathResult simulate_mean_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    SchobelZhuState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = expf(state.log_spot);
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        sum += expf(state.log_spot);
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {state, static_cast<float>(sum / observation_count)};
}

__device__ __forceinline__ SchobelZhuGeometricMeanPathResult
simulate_geometric_mean_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    SchobelZhuState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double log_sum = state.log_spot;
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        log_sum += state.log_spot;
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {
        state,
        expf(static_cast<float>(log_sum / observation_count)),
    };
}

__device__ __forceinline__ SchobelZhuTwoTimePathResult simulate_at_two_times(
    const SchobelZhuPreparedParameters& first_model,
    const SchobelZhuPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t first_num_steps,
    std::size_t second_num_steps
) {
    SchobelZhuState state = initial_state(first_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(
        first_model,
        first_num_steps,
        uniforms,
        normals,
        state
    );
    const SchobelZhuState first_state = state;
    simulate_steps(
        second_model,
        second_num_steps,
        uniforms,
        normals,
        state
    );
    return {first_state, state};
}

__device__ __forceinline__ SchobelZhuMaximumPathResult simulate_maximum_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    SchobelZhuState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = expf(state.log_spot);
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        maximum = fmaxf(maximum, expf(state.log_spot));
    }
    return {state, maximum};
}

__device__ __forceinline__ SchobelZhuState simulate_on_regular_grid(
    const SchobelZhuPreparedParameters& initial_stub_model,
    const SchobelZhuPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* observed_spots,
    float* observed_volatilities
) {
    SchobelZhuState state = initial_state(initial_stub_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(
        initial_stub_model,
        initial_stub_steps,
        uniforms,
        normals,
        state
    );
    if (exercise_count == 1U) {
        return state;
    }

    std::size_t output_index = path;
    observed_spots[output_index] = expf(state.log_spot);
    observed_volatilities[output_index] = state.volatility;
    for (std::uint32_t exercise_index = 1U;
         exercise_index + 1U < exercise_count;
         ++exercise_index) {
        simulate_steps(
            regular_model,
            steps_per_exercise,
            uniforms,
            normals,
            state
        );
        output_index += path_count;
        observed_spots[output_index] = expf(state.log_spot);
        observed_volatilities[output_index] = state.volatility;
    }
    simulate_steps(
        regular_model,
        steps_per_exercise,
        uniforms,
        normals,
        state
    );
    return state;
}

}  // namespace ai_factory::workbench::schobel_zhu
