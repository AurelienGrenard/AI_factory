// Device path-update definitions for the rough Stein--Stein Volterra dynamics.
#pragma once

#include "model/equity/rough/rough_stein_stein/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::rough_stein_stein {

__device__ __forceinline__ auto PathPolicy::kernel_parameters(
    const Parameters& parameters
) -> volterra::FractionalResolventHybridKernelPolicy::Parameters {
    return {parameters.hurst_exponent, parameters.mean_reversion};
}

__device__ __forceinline__ PathPolicy::PreparedModel PathPolicy::prepare_model(
    const Parameters& parameters,
    float time_step
) {
    return {
        logf(parameters.spot),
        parameters.volatility_level,
        parameters.volatility_of_volatility,
        time_step,
        sqrtf(time_step),
        (parameters.risk_free_rate - parameters.dividend_yield) * time_step,
        parameters.rho,
        sqrtf(fmaxf(1.0f - parameters.rho * parameters.rho, 0.0f)),
    };
}

__device__ __forceinline__ PathPolicy::State PathPolicy::initial_state(
    const PreparedModel& model
) {
    return {model.initial_log_spot, model.volatility_level};
}

__device__ __forceinline__ float map_volatility(
    const PreparedModel& model, float value, float
) {
    return model.volatility_level
        + model.volatility_of_volatility * value;
}

__device__ __forceinline__ void advance_spot_cell(
    const PreparedModel& model, float volatility, float normal,
    float& log_spot
) {
    log_spot = fmaf(
        volatility * model.sqrt_time_step, normal,
        log_spot + model.drift_time_step
            - 0.5f * volatility * volatility * model.time_step);
}

__device__ __forceinline__ void PathPolicy::advance(
    const PreparedModel& model,
    float volterra_value,
    float,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    const float spot_normal = fmaf(
        model.rho,
        rough_normal,
        model.orthogonal_correlation * independent_spot_normal
    );
    advance_spot_cell(model, state.volatility,
                      spot_normal, state.log_spot);
    state.volatility = map_volatility(model, volterra_value, 0.0f);
}

__device__ __forceinline__ float PathPolicy::spot(const State& state) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float PathPolicy::log_spot(const State& state) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
