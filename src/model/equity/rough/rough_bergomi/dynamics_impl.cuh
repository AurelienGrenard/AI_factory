// Included device definitions mapping the normalized Volterra process to rough-Bergomi state.
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

__device__ __forceinline__ float map_volatility(
    const PreparedModel& model, float value, float variance
) {
    return expf(
        model.log_initial_variance + model.eta * value
        - model.half_eta_squared * variance);
}

__device__ __forceinline__ void advance_spot_cell(
    const PreparedModel& model, float variance, float normal,
    float& log_spot
) {
    log_spot = fmaf(
        sqrtf(variance) * model.sqrt_time_step, normal,
        log_spot + model.drift_time_step
            - 0.5f * variance * model.time_step);
}

__device__ __forceinline__ void advance(
    const PreparedModel& prepared_model,
    float volterra_value,
    float volterra_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    const float correlated_normal = fmaf(
        prepared_model.rho,
        rough_normal,
        prepared_model.orthogonal_correlation * independent_spot_normal
    );
    advance_spot_cell(prepared_model, state.variance,
                      correlated_normal, state.log_spot);
    state.variance = map_volatility(
        prepared_model, volterra_value, volterra_variance);
}

__device__ __forceinline__ float PathPolicy::kernel_parameters(
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
    float volterra_value,
    float volterra_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    rough_bergomi::advance(
        prepared_model,
        volterra_value,
        volterra_variance,
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
