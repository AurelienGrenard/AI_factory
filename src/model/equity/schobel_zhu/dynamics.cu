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

}  // namespace

// ======================== Common equity dynamics =========================

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    return schobel_zhu::prepare_model(parameters, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return schobel_zhu::initial_state(dynamics);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    schobel_zhu::simulate_one_step(
        dynamics,
        random.uniforms,
        random.normals,
        state
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

__device__ __forceinline__ float DynamicsPolicy::spot(const State& state) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float DynamicsPolicy::log_spot(
    const State& state
) {
    return state.log_spot;
}

__device__ __forceinline__ float DynamicsPolicy::risk_free_rate(
    const Parameters& parameters
) {
    return parameters.risk_free_rate;
}

}  // namespace ai_factory::workbench::schobel_zhu
