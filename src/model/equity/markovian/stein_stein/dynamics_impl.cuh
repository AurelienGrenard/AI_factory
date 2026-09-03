#pragma once

#include "model/equity/markovian/stein_stein/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::stein_stein {

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    const float decay = expf(-parameters.mean_reversion * delta_t);
    const float endpoint_variance = -expm1f(
        -2.0f * parameters.mean_reversion * delta_t
    ) / (2.0f * parameters.mean_reversion);
    const float endpoint_increment_correlation = -expm1f(
        -parameters.mean_reversion * delta_t
    ) / (
        parameters.mean_reversion * sqrtf(delta_t * endpoint_variance)
    );
    const float correlation = fminf(
        fmaxf(endpoint_increment_correlation, -1.0f),
        1.0f
    );
    return {
        logf(parameters.spot),
        parameters.initial_volatility,
        decay,
        parameters.volatility_of_volatility * sqrtf(endpoint_variance),
        correlation,
        sqrtf(fmaxf(1.0f - correlation * correlation, 0.0f)),
        (parameters.risk_free_rate - parameters.dividend_yield) * delta_t,
        sqrtf(delta_t),
        parameters.rho,
        sqrtf(fmaxf(1.0f - parameters.rho * parameters.rho, 0.0f)),
    };
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return {dynamics.initial_log_spot, dynamics.initial_volatility};
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    const float endpoint_normal = philox::next_normal(
        random.uniforms, random.normals
    );
    const float increment_residual = philox::next_normal(
        random.uniforms, random.normals
    );
    const float asset_residual = philox::next_normal(
        random.uniforms, random.normals
    );
    const float volatility_increment_normal = fmaf(
        dynamics.endpoint_increment_correlation,
        endpoint_normal,
        dynamics.endpoint_increment_residual * increment_residual
    );
    const float asset_normal = fmaf(
        dynamics.rho,
        volatility_increment_normal,
        dynamics.correlation_residual * asset_residual
    );
    const float volatility = state.volatility;
    state.log_spot += dynamics.drift_dt
        - 0.5f * volatility * volatility
            * dynamics.sqrt_dt * dynamics.sqrt_dt
        + volatility * dynamics.sqrt_dt * asset_normal;
    state.volatility = volatility * dynamics.volatility_decay
        + dynamics.volatility_standard_deviation * endpoint_normal;
}

__device__ __forceinline__ void DynamicsPolicy::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        simulate_one_step(dynamics, random, state);
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

}  // namespace ai_factory::workbench::model::equity::stein_stein
