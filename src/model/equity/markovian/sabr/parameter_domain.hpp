// Shared SABR parameter domain for loading and sensitivity-node preparation.
#pragma once

#include "model/equity/markovian/sabr/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::sabr {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const ModelParameters& p
) {
    return finite_parameter(p.spot) && p.spot > 0.0f
        && finite_parameter(p.risk_free_rate)
        && finite_parameter(p.dividend_yield)
        && finite_parameter(p.initial_volatility)
        && p.initial_volatility > 0.0f
        && finite_parameter(p.volatility_of_volatility)
        && p.volatility_of_volatility >= 0.0f
        && finite_parameter(p.rho) && p.rho >= -1.0f && p.rho <= 1.0f
        && finite_parameter(p.beta) && p.beta >= 0.0f && p.beta <= 1.0f;
}

inline void validate_parameters(const ModelParameters& p, const std::string& prefix) {
    if (!valid_parameters(p)) {
        throw std::invalid_argument(prefix + "invalid model parameters.");
    }
}

}  // namespace ai_factory::workbench::model::equity::sabr
