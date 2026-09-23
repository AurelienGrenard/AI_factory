// Ornstein-Uhlenbeck parameter domain shared by host and device code.
#pragma once

#include "model/fixed_income/ornstein_uhlenbeck/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(const ModelParameters& model) {
    return finite_parameter(model.process.mean_reversion)
        && model.process.mean_reversion > 0.0f
        && finite_parameter(model.process.volatility)
        && model.process.volatility >= 0.0f
        && finite_parameter(model.initial_state);
}

inline void validate_parameters(
    const ModelParameters& model,
    const std::string& prefix
) {
    if (!std::isfinite(model.process.mean_reversion)
        || !(model.process.mean_reversion > 0.0f))
        throw std::invalid_argument(
            prefix + "mean_reversion must be finite and positive."
        );
    if (!std::isfinite(model.process.volatility)
        || !(model.process.volatility >= 0.0f))
        throw std::invalid_argument(
            prefix + "volatility must be finite and non-negative."
        );
    if (!std::isfinite(model.initial_state))
        throw std::invalid_argument(prefix + "initial_state must be finite.");
}

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
