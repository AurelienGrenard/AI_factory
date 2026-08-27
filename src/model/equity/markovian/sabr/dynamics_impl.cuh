#pragma once

#include "model/equity/markovian/sabr/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::sabr {

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    const float one_minus_beta = 1.0f - parameters.beta;
    const float dimensional_alpha = parameters.initial_volatility
        * powf(parameters.spot, one_minus_beta);
    return {
        logf(parameters.spot),
        dimensional_alpha,
        parameters.beta,
        one_minus_beta,
        delta_t,
        sqrtf(delta_t),
        (parameters.risk_free_rate - parameters.dividend_yield) * delta_t,
        parameters.volatility_of_volatility * sqrtf(delta_t),
        0.5f * parameters.volatility_of_volatility
            * parameters.volatility_of_volatility * delta_t,
        parameters.rho,
        sqrtf(fmaxf(1.0f - parameters.rho * parameters.rho, 0.0f)),
    };
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return {
        dynamics.initial_log_spot,
        dynamics.initial_alpha,
    };
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    const float alpha_normal = philox::next_normal(
        random.uniforms, random.normals
    );
    const float residual_normal = philox::next_normal(
        random.uniforms, random.normals
    );
    const float forward_normal = fmaf(
        dynamics.rho,
        alpha_normal,
        dynamics.correlation_residual * residual_normal
    );

    if (dynamics.one_minus_beta < 1.0e-4f) {
        state.log_spot +=
            dynamics.carry_time_step
            - 0.5f * state.alpha * state.alpha * dynamics.time_step
            + state.alpha * dynamics.sqrt_time_step * forward_normal;
    } else {
        const float spot = expf(state.log_spot);
        const float transformed = powf(
            spot,
            dynamics.one_minus_beta
        ) / dynamics.one_minus_beta;
        const float next_transformed = fmaxf(
            transformed
                + dynamics.one_minus_beta * transformed
                    * dynamics.carry_time_step
                + state.alpha * dynamics.sqrt_time_step * forward_normal
                - 0.5f * dynamics.beta * state.alpha * state.alpha
                    * dynamics.time_step
                    / (dynamics.one_minus_beta * transformed),
            1.0e-12f
        );
        state.log_spot = logf(
            dynamics.one_minus_beta * next_transformed
        ) / dynamics.one_minus_beta;
    }
    state.alpha *= expf(
        dynamics.volatility_of_volatility_sqrt_dt * alpha_normal
        - dynamics.half_volatility_of_volatility_squared_dt
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

}  // namespace ai_factory::workbench::model::equity::sabr
