// Device preparation of one sampled quadratic rough-Heston parameter row.
#pragma once

#include "common/volterra/fractional_kernel_approximation.hpp"
#include "model/equity/rough/quadratic_rough_heston/dynamics.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

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
            throw std::invalid_argument("Invalid QRH sample lift horizon or dt.");
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
        result.initial_log_spot = logf(model.spot);
        result.initial_feedback = model.initial_feedback;
        result.quadratic_scale = model.quadratic_scale;
        result.quadratic_shift = model.quadratic_shift;
        result.variance_floor = model.variance_floor;
        result.feedback_rate = model.feedback_rate;
        result.feedback_volatility = model.feedback_volatility;
        result.drift_dt = (model.risk_free_rate - model.dividend_yield)
            * config.dt;
        result.dt = config.dt;
        result.sqrt_dt = sqrtf(config.dt);
        result.inverse_sqrt_dt = 1.0f / result.sqrt_dt;
        for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
            const float node = kernel.nodes[factor];
            const float decay = expf(-node * config.dt);
            const float drift_integral = -expm1f(-node * config.dt) / node;
            result.factor_decay[factor] = decay;
            result.factor_drift_integral[factor] = drift_integral;
            result.feedback_cell_loading = fmaf(
                kernel.weights[factor], drift_integral,
                result.feedback_cell_loading
            );
        }
        return result;
    }
};

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
