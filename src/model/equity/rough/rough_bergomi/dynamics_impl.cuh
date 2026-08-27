// Rough-Bergomi state mapping for the normalized fractional driver.
#pragma once

#include "model/equity/rough/rough_bergomi/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::rough_bergomi {

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float time_step
) {
    return {
        logf(parameters.spot),
        parameters.xi_0,
        logf(parameters.xi_0),
        parameters.eta,
        0.5f * parameters.eta * parameters.eta,
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
        prepared_model.initial_variance,
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
    state.log_spot = fmaf(
        sqrtf(state.variance) * prepared_model.sqrt_time_step,
        correlated_normal,
        state.log_spot + prepared_model.drift_time_step
            - 0.5f * state.variance * prepared_model.time_step
    );
    state.variance = expf(
        prepared_model.log_initial_variance
        + prepared_model.eta * driver_value
        - prepared_model.half_eta_squared * driver_variance
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
    return rough_bergomi::prepare_model(parameters, time_step);
}

__device__ __forceinline__ PathPolicy::State PathPolicy::initial_state(
    const PreparedModel& prepared_model
) {
    return rough_bergomi::initial_state(prepared_model);
}

__device__ __forceinline__ void PathPolicy::advance(
    const PreparedModel& prepared_model,
    float driver_value,
    float driver_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    rough_bergomi::advance(
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

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
