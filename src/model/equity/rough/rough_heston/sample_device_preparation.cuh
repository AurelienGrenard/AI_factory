// Device preparation of one sampled rough-Heston parameter row.
#pragma once

#include "model/equity/rough/rough_heston/small_matrix_exponential.cuh"
#include "volterra/fractional_kernel_approximation.hpp"
#include "model/equity/rough/rough_heston/dynamics.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::rough_heston {

template<std::size_t FactorCount>
struct SampleDevicePreparer {
    using Prepared = PreparedDynamics<FactorCount>;
    using Kernel = volterra::ExponentialKernel<FactorCount>;
    struct Configuration {
        float approximation_horizon;
        float dt;
    };

    static void validate(const Configuration& config) {
        if (!(config.approximation_horizon > config.dt)
            || !(config.dt > 0.0f)) {
            throw std::invalid_argument("Invalid RH sample lift horizon or dt.");
        }
    }

    __device__ __forceinline__ static Prepared prepare(
        const ModelParameters& model,
        const Configuration& config
    ) {
        const Kernel kernel = volterra::fit_positive_fractional_kernel_l2_device<
            FactorCount
        >(
            model.hurst_exponent, config.approximation_horizon, config.dt
        );
        Prepared result{};
        result.kernel = kernel;
        double inverse_node_weight_sum = 0.0;
        double weight_sum = 0.0;
        for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
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

        constexpr std::size_t dimension = FactorCount + 1U;
        math::SmallMatrix<dimension> augmented{};
        const double half_dt = 0.5 * static_cast<double>(config.dt);
        for (std::size_t row = 0U; row < FactorCount; ++row) {
            for (std::size_t column = 0U; column < FactorCount; ++column) {
                double coefficient = -static_cast<double>(model.mean_reversion)
                    * kernel.weights[column];
                if (row == column) coefficient -= kernel.nodes[row];
                augmented[row * dimension + column] = half_dt * coefficient;
            }
            augmented[row * dimension + FactorCount] = half_dt * (
                kernel.nodes[row] * result.initial_factors[row]
                + model.variance_drift
            );
        }
        const auto exponential = math::matrix_exponential<dimension>(augmented);
        for (std::size_t row = 0U; row < FactorCount; ++row) {
            for (std::size_t column = 0U; column < FactorCount; ++column) {
                result.ode_half_step[row * FactorCount + column] =
                    static_cast<float>(exponential[row * dimension + column]);
            }
            result.ode_half_shift[row] = static_cast<float>(
                exponential[row * dimension + FactorCount]
            );
        }

        const float one_minus_rho_squared = fmaxf(
            1.0f - model.rho * model.rho, 0.0f
        );
        result.initial_log_spot = logf(model.spot);
        result.orthogonal_variance_scale = one_minus_rho_squared
            * 0.5f * config.dt;
        result.orthogonal_drift_scale = 0.25f * one_minus_rho_squared
            * config.dt;
        result.weight_sum = static_cast<float>(weight_sum);
        result.weak_variance_scale = result.weight_sum * result.weight_sum
            * model.volatility_of_variance
            * model.volatility_of_variance * config.dt;
        result.correlated_constant = -(
            kernel.nodes[0] * result.initial_factors[0]
            + model.variance_drift
        ) * config.dt;
        result.half_dt_first_node = 0.5f * config.dt * kernel.nodes[0];
        result.correlated_weight_scale = 0.5f * config.dt * (
            model.mean_reversion
            - 0.5f * model.rho * model.volatility_of_variance
        );
        result.rho_over_volatility = model.rho / model.volatility_of_variance;
        result.spot_drift_dt =
            (model.risk_free_rate - model.dividend_yield) * config.dt;
        return result;
    }
};

}  // namespace ai_factory::workbench::model::equity::rough_heston
