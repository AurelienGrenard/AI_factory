// Heston host/device parameter-domain contract and diagnostics without a Feller restriction.
#pragma once

#include "model/equity/markovian/heston/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::heston {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const ModelParameters& parameters
) {
    return finite_parameter(parameters.spot) && parameters.spot > 0.0f
        && finite_parameter(parameters.risk_free_rate)
        && finite_parameter(parameters.dividend_yield)
        && finite_parameter(parameters.initial_variance)
        && parameters.initial_variance >= 0.0f
        && finite_parameter(parameters.kappa) && parameters.kappa > 0.0f
        && finite_parameter(parameters.theta) && parameters.theta > 0.0f
        && finite_parameter(parameters.gamma) && parameters.gamma > 0.0f
        && finite_parameter(parameters.rho)
        && parameters.rho >= -1.0f && parameters.rho <= 1.0f;
}

inline const char* parameter_domain_error(const ModelParameters& parameters) {
    if (valid_parameters(parameters)) return nullptr;
    if (!std::isfinite(parameters.spot) || !(parameters.spot > 0.0f))
        return "spot must be finite and positive.";
    if (!std::isfinite(parameters.risk_free_rate))
        return "risk_free_rate must be finite.";
    if (!std::isfinite(parameters.dividend_yield))
        return "dividend_yield must be finite.";
    if (!std::isfinite(parameters.initial_variance)
        || !(parameters.initial_variance >= 0.0f))
        return "initial_variance must be finite and non-negative.";
    if (!std::isfinite(parameters.kappa) || !(parameters.kappa > 0.0f))
        return "kappa must be finite and positive.";
    if (!std::isfinite(parameters.theta) || !(parameters.theta > 0.0f))
        return "theta must be finite and positive.";
    if (!std::isfinite(parameters.gamma) || !(parameters.gamma > 0.0f))
        return "gamma must be finite and positive.";
    if (!std::isfinite(parameters.rho)
        || !(parameters.rho >= -1.0f && parameters.rho <= 1.0f))
        return "rho must be finite and lie in [-1, 1].";
    return nullptr;
}

inline void validate_parameters(
    const ModelParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::model::equity::heston
