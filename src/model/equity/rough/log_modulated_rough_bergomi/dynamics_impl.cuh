// Device path-update definitions for the log-modulated rough Bergomi Volterra dynamics.
#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

__device__ __forceinline__ auto PathPolicy::kernel_parameters(
    const Parameters& parameters
) -> volterra::LogModulatedHybridKernelPolicy::Parameters {
    return {
        parameters.hurst_exponent,
        parameters.log_modulation_scale,
        parameters.log_modulation_power,
    };
}

__device__ __forceinline__ PathPolicy::PreparedModel PathPolicy::prepare_model(
    const Parameters& parameters,
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

__device__ __forceinline__ PathPolicy::State PathPolicy::initial_state(
    const PreparedModel& model
) {
    return {model.initial_log_spot, model.initial_variance};
}

__device__ __forceinline__ void PathPolicy::advance(
    const PreparedModel& model,
    float volterra_value,
    float volterra_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
) {
    const float spot_normal = fmaf(
        model.rho,
        rough_normal,
        model.orthogonal_correlation * independent_spot_normal
    );
    state.log_spot = fmaf(
        sqrtf(state.variance) * model.sqrt_time_step,
        spot_normal,
        state.log_spot + model.drift_time_step
            - 0.5f * state.variance * model.time_step
    );
    state.variance = expf(
        model.log_initial_variance + model.eta * volterra_value
        - model.half_eta_squared * volterra_variance
    );
}

__device__ __forceinline__ float PathPolicy::spot(const State& state) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float PathPolicy::log_spot(const State& state) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
