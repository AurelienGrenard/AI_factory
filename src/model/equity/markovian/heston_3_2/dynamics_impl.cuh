#pragma once

#include "model/equity/markovian/heston_3_2/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::heston_3_2 {

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    const float volvol_squared = parameters.volatility_of_variance
        * parameters.volatility_of_variance;
    return {
        logf(parameters.spot),
        1.0f / parameters.initial_variance,
        (parameters.mean_reversion + volvol_squared) * delta_t,
        parameters.mean_reversion * parameters.long_run_variance * delta_t,
        parameters.volatility_of_variance * sqrtf(delta_t),
        0.25f * volvol_squared * delta_t,
        (parameters.risk_free_rate - parameters.dividend_yield) * delta_t,
        sqrtf(delta_t),
        parameters.rho,
        sqrtf(fmaxf(1.0f - parameters.rho * parameters.rho, 0.0f)),
    };
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return {dynamics.initial_log_spot, dynamics.initial_reciprocal_variance};
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    const float variance_normal = philox::next_normal(
        random.uniforms, random.normals
    );
    const float residual_normal = philox::next_normal(
        random.uniforms, random.normals
    );
    const float stock_normal = fmaf(
        dynamics.rho,
        variance_normal,
        dynamics.correlation_residual * residual_normal
    );
    const float reciprocal = fmaxf(state.reciprocal_variance, 1.0e-10f);
    const float variance = 1.0f / reciprocal;
    state.log_spot += dynamics.drift_dt
        - 0.5f * variance * dynamics.sqrt_dt * dynamics.sqrt_dt
        + sqrtf(variance) * dynamics.sqrt_dt * stock_normal;

    // U=1/V is a CIR process driven by -W^V.  A Milstein step retains
    // positivity much better than stepping the super-linear V diffusion.
    const float reciprocal_normal = -variance_normal;
    state.reciprocal_variance = fmaxf(
        reciprocal
            + dynamics.reciprocal_constant_drift_dt
            - dynamics.reciprocal_linear_drift_dt * reciprocal
            + dynamics.volatility_of_variance_sqrt_dt
                * sqrtf(reciprocal) * reciprocal_normal
            + dynamics.quarter_volatility_of_variance_squared_dt
                * (reciprocal_normal * reciprocal_normal - 1.0f),
        1.0e-10f
    );
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

}  // namespace ai_factory::workbench::model::equity::heston_3_2
