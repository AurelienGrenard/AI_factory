// Hull-White parameter domain shared by host loaders and device preparation.
#pragma once

#include "model/fixed_income/hull_white/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::fixed_income::hull_white {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(const ModelParameters& model) {
    return finite_parameter(model.mean_reversion)
        && model.mean_reversion > 0.0f
        && finite_parameter(model.volatility)
        && model.volatility >= 0.0f;
}

inline void validate_parameters(
    const ModelParameters& model,
    const std::string& prefix
) {
    if (!std::isfinite(model.mean_reversion)
        || !(model.mean_reversion > 0.0f))
        throw std::invalid_argument(
            prefix + "mean_reversion must be finite and positive."
        );
    if (!std::isfinite(model.volatility) || !(model.volatility >= 0.0f))
        throw std::invalid_argument(
            prefix + "volatility must be finite and non-negative."
        );
}

}  // namespace ai_factory::workbench::model::fixed_income::hull_white
