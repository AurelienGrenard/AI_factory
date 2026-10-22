// Included device definitions mapping the normalized Volterra process to rough-SABR state.
#pragma once

#include "equity/absorbing_lamperti.cuh"
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

__device__ __forceinline__ float map_volatility(
    const PreparedModel& model, float value, float variance
) {
    return expf(
        model.log_initial_volatility + model.half_eta * value
        - model.quarter_eta_squared * variance);
}

__device__ __forceinline__ void advance_spot_cell(
    const PreparedModel& model, float volatility, float normal,
    float& log_spot
) {
    if (model.one_minus_beta < 1.0e-4f) {
        log_spot += model.drift_time_step
            - 0.5f * volatility * volatility * model.time_step
            + volatility * model.sqrt_time_step * normal;
    } else {
        ::ai_factory::workbench::equity::advance_absorbing_lamperti_log_spot(
            model.beta, model.one_minus_beta, model.drift_time_step,
            model.time_step, model.sqrt_time_step, volatility, normal,
            log_spot);
    }
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
    advance_spot_cell(prepared_model, state.volatility,
                      correlated_normal, state.log_spot);
    state.volatility = map_volatility(
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
    return rough_sabr::prepare_model(parameters, time_step);
}

__device__ __forceinline__ PathPolicy::State PathPolicy::initial_state(
    const PreparedModel& prepared_model
) {
    return rough_sabr::initial_state(prepared_model);
}

__device__ __forceinline__ void PathPolicy::advance(
    const PreparedModel& prepared_model,
    float volterra_value,
    float volterra_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    rough_sabr::advance(
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

}  // namespace ai_factory::workbench::model::equity::rough_sabr
