// Host preparation of fixed-factor rough-Heston dynamics.
#pragma once

#include "common/volterra/fractional_kernel_approximation.hpp"
#include "model/equity/rough/rough_heston/dynamics.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ai_factory::workbench::model::equity::rough_heston {
namespace detail {

template<std::size_t Dimension>
using Matrix = std::array<double, Dimension * Dimension>;

template<std::size_t Dimension>
Matrix<Dimension> identity_matrix() {
    Matrix<Dimension> result{};
    for (std::size_t index = 0U; index < Dimension; ++index)
        result[index * Dimension + index] = 1.0;
    return result;
}

template<std::size_t Dimension>
double matrix_one_norm(const Matrix<Dimension>& matrix) {
    double norm = 0.0;
    for (std::size_t column = 0U; column < Dimension; ++column) {
        double sum = 0.0;
        for (std::size_t row = 0U; row < Dimension; ++row)
            sum += std::abs(matrix[row * Dimension + column]);
        norm = std::max(norm, sum);
    }
    return norm;
}

template<std::size_t Dimension>
Matrix<Dimension> multiply(
    const Matrix<Dimension>& left,
    const Matrix<Dimension>& right
) {
    Matrix<Dimension> result{};
    for (std::size_t row = 0U; row < Dimension; ++row) {
        for (std::size_t column = 0U; column < Dimension; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0U; inner < Dimension; ++inner) {
                value += left[row * Dimension + inner]
                    * right[inner * Dimension + column];
            }
            result[row * Dimension + column] = value;
        }
    }
    return result;
}

// Scaling/squaring with a converged Taylor series is ample for Dimension<=8
// and runs only during host preparation. It also handles the augmented affine
// system without a potentially ill-conditioned explicit matrix inverse.
template<std::size_t Dimension>
Matrix<Dimension> matrix_exponential(Matrix<Dimension> matrix) {
    const double norm = matrix_one_norm<Dimension>(matrix);
    int scaling = 0;
    if (norm > 0.5)
        scaling = static_cast<int>(std::ceil(std::log2(norm / 0.5)));
    const double divisor = std::ldexp(1.0, scaling);
    for (double& value : matrix) value /= divisor;

    Matrix<Dimension> result = identity_matrix<Dimension>();
    Matrix<Dimension> term = result;
    for (int order = 1; order <= 64; ++order) {
        term = multiply<Dimension>(term, matrix);
        const double inverse_order = 1.0 / static_cast<double>(order);
        for (double& value : term) value *= inverse_order;
        for (std::size_t index = 0U; index < result.size(); ++index)
            result[index] += term[index];
        if (matrix_one_norm<Dimension>(term) < 2.0e-16) break;
    }
    for (int square = 0; square < scaling; ++square)
        result = multiply<Dimension>(result, result);
    return result;
}

inline void validate_model(const ModelParameters& model) {
    if (!std::isfinite(model.spot) || !(model.spot > 0.0f)
        || !std::isfinite(model.initial_variance)
        || !(model.initial_variance > 0.0f)
        || !std::isfinite(model.mean_reversion)
        || !(model.mean_reversion >= 0.0f)
        || !std::isfinite(model.variance_drift)
        || !(model.variance_drift >= 0.0f)
        || !std::isfinite(model.volatility_of_variance)
        || !(model.volatility_of_variance > 0.0f)
        || !std::isfinite(model.hurst_exponent)
        || !(model.hurst_exponent > 0.0f
             && model.hurst_exponent < 0.5f)
        || !std::isfinite(model.rho)
        || !(model.rho >= -1.0f && model.rho <= 1.0f)
        || !std::isfinite(model.risk_free_rate)
        || !std::isfinite(model.dividend_yield)) {
        throw std::invalid_argument("Invalid rough-Heston model parameters.");
    }
}

}  // namespace detail

template<std::size_t FactorCount>
PreparedDynamics<FactorCount> prepare_dynamics(
    const ModelParameters& model,
    const volterra::ExponentialKernel<FactorCount>& kernel,
    float approximation_horizon,
    float dt
) {
    detail::validate_model(model);
    if (!std::isfinite(approximation_horizon)
        || !(approximation_horizon > 0.0f)
        || !std::isfinite(dt) || !(dt > 0.0f)) {
        throw std::invalid_argument(
            "rough-Heston horizon and dt must be finite and positive."
        );
    }

    PreparedDynamics<FactorCount> result{};
    result.kernel = kernel;
    double inverse_node_weight_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        if (!std::isfinite(kernel.nodes[factor])
            || !(kernel.nodes[factor] > 0.0f)
            || !std::isfinite(kernel.weights[factor])
            || !(kernel.weights[factor] > 0.0f)) {
            throw std::invalid_argument(
                "rough-Heston exponential nodes and weights must be positive."
            );
        }
        inverse_node_weight_sum +=
            kernel.weights[factor] / kernel.nodes[factor];
        weight_sum += kernel.weights[factor];
    }
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        result.initial_factors[factor] = static_cast<float>(
            model.initial_variance / kernel.nodes[factor]
            / inverse_node_weight_sum
        );
    }

    constexpr std::size_t augmented_dimension = FactorCount + 1U;
    detail::Matrix<augmented_dimension> augmented{};
    const double half_dt = 0.5 * static_cast<double>(dt);
    for (std::size_t row = 0U; row < FactorCount; ++row) {
        for (std::size_t column = 0U; column < FactorCount; ++column) {
            double coefficient = -static_cast<double>(model.mean_reversion)
                * kernel.weights[column];
            if (row == column) coefficient -= kernel.nodes[row];
            augmented[row * augmented_dimension + column] =
                half_dt * coefficient;
        }
        augmented[row * augmented_dimension + FactorCount] = half_dt * (
            kernel.nodes[row] * result.initial_factors[row]
            + model.variance_drift
        );
    }
    const auto exponential = detail::matrix_exponential<
        augmented_dimension
    >(augmented);
    for (std::size_t row = 0U; row < FactorCount; ++row) {
        for (std::size_t column = 0U; column < FactorCount; ++column) {
            result.ode_half_step[row * FactorCount + column] =
                static_cast<float>(
                    exponential[row * augmented_dimension + column]
                );
        }
        result.ode_half_shift[row] = static_cast<float>(
            exponential[row * augmented_dimension + FactorCount]
        );
    }

    const float one_minus_rho_squared =
        std::max(1.0f - model.rho * model.rho, 0.0f);
    result.initial_log_spot = std::log(model.spot);
    result.orthogonal_variance_scale = one_minus_rho_squared * 0.5f * dt;
    result.orthogonal_drift_scale = 0.25f * one_minus_rho_squared * dt;
    result.weight_sum = static_cast<float>(weight_sum);
    result.weak_variance_scale = result.weight_sum * result.weight_sum
        * model.volatility_of_variance * model.volatility_of_variance * dt;
    result.correlated_constant = -(
        kernel.nodes[0] * result.initial_factors[0]
        + model.variance_drift
    ) * dt;
    result.half_dt_first_node = 0.5f * dt * kernel.nodes[0];
    result.correlated_weight_scale = 0.5f * dt * (
        model.mean_reversion
        - 0.5f * model.rho * model.volatility_of_variance
    );
    result.rho_over_volatility =
        model.rho / model.volatility_of_variance;
    result.spot_drift_dt =
        (model.risk_free_rate - model.dividend_yield) * dt;
    return result;
}

template<std::size_t FactorCount>
PreparedDynamics<FactorCount> prepare_dynamics(
    const ModelParameters& model,
    float approximation_horizon,
    float dt
) {
    const auto kernel = volterra::fit_positive_fractional_kernel_l2<
        FactorCount
    >(model.hurst_exponent, approximation_horizon, dt);
    return prepare_dynamics(model, kernel, approximation_horizon, dt);
}

template<std::size_t FactorCount>
std::vector<PreparedDynamics<FactorCount>> prepare_dynamics(
    const std::vector<ModelParameters>& models,
    float approximation_horizon,
    float dt
) {
    std::vector<PreparedDynamics<FactorCount>> result;
    result.reserve(models.size());
    std::unordered_map<
        std::uint32_t,
        volterra::ExponentialKernel<FactorCount>
    > kernels_by_hurst;
    for (const ModelParameters& model : models) {
        const std::uint32_t hurst_bits = std::bit_cast<std::uint32_t>(
            model.hurst_exponent
        );
        auto kernel = kernels_by_hurst.find(hurst_bits);
        if (kernel == kernels_by_hurst.end()) {
            kernel = kernels_by_hurst.emplace(
                hurst_bits,
                volterra::fit_positive_fractional_kernel_l2<FactorCount>(
                    model.hurst_exponent, approximation_horizon, dt
                )
            ).first;
        }
        result.push_back(prepare_dynamics(
            model,
            kernel->second,
            approximation_horizon,
            dt
        ));
    }
    return result;
}

}  // namespace ai_factory::workbench::model::equity::rough_heston
