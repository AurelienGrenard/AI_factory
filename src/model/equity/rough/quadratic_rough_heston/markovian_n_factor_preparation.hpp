// Host-side quadrature and coefficient preparation for quadratic rough Heston N-factor lifts.
#pragma once

#include "common/volterra/fractional_kernel_approximation.hpp"
#include "model/equity/rough/quadratic_rough_heston/dynamics.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {
namespace detail {

inline void validate_model(const ModelParameters& model) {
    if (!std::isfinite(model.spot) || !(model.spot > 0.0f)
        || !std::isfinite(model.risk_free_rate)
        || !std::isfinite(model.dividend_yield)
        || !std::isfinite(model.initial_feedback)
        || !std::isfinite(model.quadratic_scale)
        || !(model.quadratic_scale > 0.0f)
        || !std::isfinite(model.quadratic_shift)
        || !std::isfinite(model.variance_floor)
        || !(model.variance_floor > 0.0f)
        || !std::isfinite(model.feedback_rate)
        || !(model.feedback_rate > 0.0f)
        || !std::isfinite(model.feedback_volatility)
        || !(model.feedback_volatility > 0.0f)
        || !std::isfinite(model.hurst_exponent)
        || !(model.hurst_exponent > 0.0f
             && model.hurst_exponent < 0.5f)) {
        throw std::invalid_argument(
            "Invalid quadratic rough-Heston model parameters."
        );
    }
}

template<std::size_t FactorCount>
void validate_kernel(
    const volterra::ExponentialKernel<FactorCount>& kernel
) {
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        if (!std::isfinite(kernel.nodes[factor])
            || !(kernel.nodes[factor] > 0.0f)
            || !std::isfinite(kernel.weights[factor])
            || !(kernel.weights[factor] > 0.0f)) {
            throw std::invalid_argument(
                "Quadratic rough-Heston exponential nodes and weights "
                "must be finite and positive."
            );
        }
    }
}

template<std::size_t FactorCount>
void validate_prepared_dynamics(
    const PreparedDynamics<FactorCount>& prepared
) {
    if (!std::isfinite(prepared.initial_log_spot)
        || !std::isfinite(prepared.drift_dt)
        || !std::isfinite(prepared.dt) || !(prepared.dt > 0.0f)
        || !std::isfinite(prepared.sqrt_dt) || !(prepared.sqrt_dt > 0.0f)
        || !std::isfinite(prepared.inverse_sqrt_dt)
        || !(prepared.inverse_sqrt_dt > 0.0f)
        || !std::isfinite(prepared.feedback_cell_loading)
        || !(prepared.feedback_cell_loading > 0.0f)
        || !std::isfinite(
            prepared.feedback_rate * prepared.feedback_volatility
                * prepared.inverse_sqrt_dt
        )) {
        throw std::overflow_error(
            "Quadratic rough-Heston preparation produced non-finite "
            "coefficients."
        );
    }
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        if (!std::isfinite(prepared.factor_decay[factor])
            || prepared.factor_decay[factor] < 0.0f
            || prepared.factor_decay[factor] > 1.0f
            || !std::isfinite(prepared.factor_drift_integral[factor])
            || !(prepared.factor_drift_integral[factor] > 0.0f)) {
            throw std::overflow_error(
                "Quadratic rough-Heston preparation produced an invalid "
                "factor coefficient."
            );
        }
    }
}

}  // namespace detail

template<std::size_t FactorCount>
PreparedDynamics<FactorCount> prepare_dynamics(
    const ModelParameters& model,
    const volterra::ExponentialKernel<FactorCount>& kernel,
    float dt
) {
    detail::validate_model(model);
    detail::validate_kernel(kernel);
    if (!std::isfinite(dt) || !(dt > 0.0f)) {
        throw std::invalid_argument(
            "Quadratic rough-Heston dt must be finite and positive."
        );
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
    detail::validate_prepared_dynamics(result);
    return result;
}

template<std::size_t FactorCount>
PreparedDynamics<FactorCount> prepare_dynamics(
    const ModelParameters& model,
    float approximation_horizon,
    float dt
) {
    detail::validate_model(model);
    auto kernel = volterra::fit_positive_fractional_kernel_l2<FactorCount>(
        model.hurst_exponent,
        approximation_horizon,
        dt
    );
    return prepare_dynamics(model, kernel, dt);
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

template<std::size_t FactorCount, std::size_t HurstGridPointCount>
std::vector<PreparedDynamics<FactorCount>> prepare_dynamics_on_hurst_grid(
    const std::vector<ModelParameters>& models,
    float approximation_horizon,
    float dt,
    float minimum_hurst_exponent,
    float maximum_hurst_exponent
) {
    const auto grid =
        volterra::fit_positive_fractional_kernel_l2_hurst_grid<
            FactorCount,
            HurstGridPointCount
        >(
            minimum_hurst_exponent,
            maximum_hurst_exponent,
            approximation_horizon,
            dt
        );
    std::vector<PreparedDynamics<FactorCount>> prepared;
    prepared.reserve(models.size());
    for (const ModelParameters& model : models) {
        prepared.push_back(prepare_dynamics(
            model,
            grid.interpolate(model.hurst_exponent),
            dt
        ));
    }
    return prepared;
}

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
