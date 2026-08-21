#include "model/equity/schobel_zhu/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::schobel_zhu {

// ======================== Common equity dynamics =========================

// Prepare the coefficients defining one transition of duration delta_t
// under the supplied model parameters.
__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float delta_t
) {
    const float sqrt_dt = sqrtf(delta_t);
    const float exp_mean_reversion_dt = expf(
        -parameters.mean_reversion * delta_t
    );
    const float endpoint_variance =
        (1.0f - exp_mean_reversion_dt * exp_mean_reversion_dt)
        / (2.0f * parameters.mean_reversion);
    const float endpoint_increment_correlation =
        (1.0f - exp_mean_reversion_dt)
        / (
            parameters.mean_reversion
            * sqrtf(delta_t * endpoint_variance)
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
        (parameters.risk_free_rate - parameters.dividend_yield) * delta_t,
        sqrt_dt,
        parameters.correlation,
        sqrtf(1.0f - parameters.correlation * parameters.correlation),
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return {prepared_model.initial_log_spot, prepared_model.initial_volatility};
}

// The OU endpoint is exact. Its Gaussian innovation is coupled to the
// volatility Brownian increment used by the log-spot Euler step, preserving
// the requested instantaneous asset/volatility correlation.
__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model,
    float ou_normal,
    float increment_residual_normal,
    float asset_residual_normal,
    State& state
) {
    const float volatility_brownian_normal = fmaf(
        prepared_model.endpoint_increment_residual,
        increment_residual_normal,
        prepared_model.endpoint_increment_correlation * ou_normal
    );
    const float asset_normal = fmaf(
        prepared_model.correlation_residual,
        asset_residual_normal,
        prepared_model.correlation * volatility_brownian_normal
    );
    const float volatility = state.volatility;
    state.log_spot +=
        prepared_model.drift_dt
        - 0.5f * volatility * volatility * prepared_model.sqrt_dt * prepared_model.sqrt_dt
        + volatility * prepared_model.sqrt_dt * asset_normal;
    state.volatility =
        prepared_model.long_run_volatility
        + (volatility - prepared_model.long_run_volatility)
            * prepared_model.exp_mean_reversion_dt
        + prepared_model.ou_std * ou_normal;
}

// ==================== Model-specific implementation =======================

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& prepared_model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
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
        prepared_model,
        ou_normal,
        increment_residual_normal,
        asset_residual_normal,
        state
    );
}

__device__ __forceinline__ void simulate_steps(
    const PreparedModel& prepared_model,
    std::uint32_t num_steps,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) {
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
    }
}

}  // namespace

// ======================== Common equity dynamics =========================

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(prepared_model, num_steps, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ MeanPathResult simulate_mean_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = expf(state.log_spot);
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
        sum += expf(state.log_spot);
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {static_cast<float>(sum / observation_count)};
}

__device__ __forceinline__ GeometricMeanPathResult
simulate_geometric_mean_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double log_sum = state.log_spot;
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
        log_sum += state.log_spot;
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {
        expf(static_cast<float>(log_sum / observation_count)),
    };
}

__device__ __forceinline__ MaximumPathResult simulate_maximum_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = expf(state.log_spot);
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
        maximum = fmaxf(maximum, expf(state.log_spot));
    }
    return {maximum};
}

__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_volatilities
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(
        prepared_model,
        initial_stub_steps,
        uniforms,
        normals,
        state
    );
    if (observation_count == 1U) {
        return state;
    }

    std::size_t output_index = 0U;
    observed_spots[output_index] = expf(state.log_spot);
    observed_volatilities[output_index] = state.volatility;
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_steps(
            prepared_model,
            steps_per_observation,
            uniforms,
            normals,
            state
        );
        output_index += observation_stride;
        observed_spots[output_index] = expf(state.log_spot);
        observed_volatilities[output_index] = state.volatility;
    }
    simulate_steps(
        prepared_model,
        steps_per_observation,
        uniforms,
        normals,
        state
    );
    return state;
}

__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    const std::uint32_t* __restrict__ steps_between_observations,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_volatilities
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    std::size_t output_index = 0U;
    for (std::uint32_t observation = 0U;
         observation < observation_count;
         ++observation) {
        simulate_steps(
            prepared_model,
            steps_between_observations[observation],
            uniforms,
            normals,
            state
        );
        if (observation + 1U < observation_count) {
            observed_spots[output_index] = expf(state.log_spot);
            observed_volatilities[output_index] = state.volatility;
            output_index += observation_stride;
        }
    }
    return state;
}

}  // namespace ai_factory::workbench::schobel_zhu
