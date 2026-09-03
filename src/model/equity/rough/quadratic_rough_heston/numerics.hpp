#pragma once

#include "common/volterra/fractional_kernel_approximation.hpp"
#include "model/equity/rough/quadratic_rough_heston/dynamics.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

template<std::size_t FactorCount>
PreparedDynamics<FactorCount> prepare_dynamics(
    const ModelParameters& model,
    const volterra::ExponentialKernel<FactorCount>& kernel,
    float,
    float dt
) {
    if (!(std::isfinite(model.spot) && model.spot > 0.0f)
        || !(std::isfinite(model.quadratic_scale)
             && model.quadratic_scale > 0.0f)
        || !(std::isfinite(model.variance_floor)
             && model.variance_floor > 0.0f)
        || !(std::isfinite(model.feedback_rate)
             && model.feedback_rate > 0.0f)
        || !(std::isfinite(model.feedback_volatility)
             && model.feedback_volatility > 0.0f)
        || !(std::isfinite(dt) && dt > 0.0f)) {
        throw std::invalid_argument("Invalid quadratic rough-Heston inputs.");
    }
    PreparedDynamics<FactorCount> result{};
    result.kernel = kernel;
    result.initial_log_spot = std::log(model.spot);
    result.initial_feedback = model.initial_feedback;
    result.quadratic_scale = model.quadratic_scale;
    result.quadratic_shift = model.quadratic_shift;
    result.variance_floor = model.variance_floor;
    result.feedback_rate = model.feedback_rate;
    result.feedback_volatility = model.feedback_volatility;
    result.drift_dt = (model.risk_free_rate - model.dividend_yield) * dt;
    result.dt = dt;
    result.sqrt_dt = std::sqrt(dt);
    result.inverse_sqrt_dt = 1.0f / result.sqrt_dt;
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        const float node = kernel.nodes[factor];
        const float decay = std::exp(-node * dt);
        const float drift_integral = -std::expm1(-node * dt) / node;
        result.factor_decay[factor] = decay;
        result.factor_drift_integral[factor] = drift_integral;
        result.feedback_cell_loading = std::fma(
            kernel.weights[factor],
            drift_integral,
            result.feedback_cell_loading
        );
    }
    return result;
}

template<std::size_t FactorCount>
PreparedDynamics<FactorCount> prepare_dynamics(
    const ModelParameters& model,
    float approximation_horizon,
    float dt
) {
    auto kernel = volterra::fit_positive_fractional_kernel_l2<FactorCount>(
        model.hurst_exponent,
        approximation_horizon,
        dt
    );
    return prepare_dynamics(model, kernel, approximation_horizon, dt);
}

template<std::size_t FactorCount>
std::vector<PreparedDynamics<FactorCount>> prepare_dynamics(
    const std::vector<ModelParameters>& models,
    float approximation_horizon,
    float dt
) {
    std::vector<PreparedDynamics<FactorCount>> prepared;
    prepared.reserve(models.size());
    for (const ModelParameters& model : models) {
        prepared.push_back(prepare_dynamics<FactorCount>(
            model,
            approximation_horizon,
            dt
        ));
    }
    return prepared;
}

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
