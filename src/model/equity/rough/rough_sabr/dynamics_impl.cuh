// Rough-SABR state mapping for the normalized fractional driver.
#pragma once

#include "model/equity/rough/rough_sabr/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::rough_sabr {

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float time_step
) {
    const float one_minus_beta = 1.0f - parameters.beta;
    const float initial_volatility = sqrtf(parameters.xi_0)
        * powf(parameters.spot, one_minus_beta);
    return {
        logf(parameters.spot),
        initial_volatility,
        logf(initial_volatility),
        0.5f * parameters.eta,
        0.25f * parameters.eta * parameters.eta,
        parameters.beta,
        one_minus_beta,
        time_step,
        sqrtf(time_step),
        (parameters.risk_free_rate - parameters.dividend_yield) * time_step,
        parameters.rho,
        sqrtf(fmaxf(1.0f - parameters.rho * parameters.rho, 0.0f)),
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return {
        prepared_model.initial_log_spot,
        prepared_model.initial_volatility,
    };
}

__device__ __forceinline__ void advance(
    const PreparedModel& prepared_model,
    float driver_value,
    float driver_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    const float correlated_normal = fmaf(
        prepared_model.rho,
        rough_normal,
        prepared_model.orthogonal_correlation * independent_spot_normal
    );
    if (prepared_model.one_minus_beta < 1.0e-4f) {
        state.log_spot +=
            prepared_model.drift_time_step
            - 0.5f * state.volatility * state.volatility
                * prepared_model.time_step
            + state.volatility * prepared_model.sqrt_time_step
                * correlated_normal;
    } else {
        const float spot = expf(state.log_spot);
        const float transformed = powf(
            spot,
            prepared_model.one_minus_beta
        ) / prepared_model.one_minus_beta;
        const float next_transformed = fmaxf(
            transformed
                + prepared_model.one_minus_beta * transformed
                    * prepared_model.drift_time_step
                + state.volatility * prepared_model.sqrt_time_step
                    * correlated_normal
                - 0.5f * prepared_model.beta
                    * state.volatility * state.volatility
                    * prepared_model.time_step
                    / (prepared_model.one_minus_beta * transformed),
            1.0e-12f
        );
        state.log_spot = logf(
            prepared_model.one_minus_beta * next_transformed
        ) / prepared_model.one_minus_beta;
    }
    state.volatility = expf(
        prepared_model.log_initial_volatility
        + prepared_model.half_eta * driver_value
        - prepared_model.quarter_eta_squared * driver_variance
    );
}

__device__ __forceinline__ float PathPolicy::driver_parameters(
    const Parameters& parameters
) {
    return parameters.hurst_exponent;
}

__device__ __forceinline__ PathPolicy::PreparedModel
PathPolicy::prepare_model(
    const Parameters& parameters,
    float time_step
) {
    return rough_sabr::prepare_model(parameters, time_step);
}

__device__ __forceinline__ PathPolicy::State PathPolicy::initial_state(
    const PreparedModel& prepared_model
) {
    return rough_sabr::initial_state(prepared_model);
}

__device__ __forceinline__ void PathPolicy::advance(
    const PreparedModel& prepared_model,
    float driver_value,
    float driver_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    rough_sabr::advance(
        prepared_model,
        driver_value,
        driver_variance,
        rough_normal,
        independent_spot_normal,
        state
    );
}

__device__ __forceinline__ float PathPolicy::spot(const State& state) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float PathPolicy::log_spot(const State& state) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::model::equity::rough_sabr
