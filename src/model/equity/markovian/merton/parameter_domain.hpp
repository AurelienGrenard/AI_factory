// Merton parameter-domain predicate and host validation diagnostics.
#pragma once

#include "model/equity/markovian/merton/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::merton {

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
        && finite_parameter(parameters.volatility)
        && parameters.volatility > 0.0f
        && finite_parameter(parameters.jump_intensity)
        && parameters.jump_intensity >= 0.0f
        && finite_parameter(parameters.jump_log_mean)
        && finite_parameter(parameters.jump_log_volatility)
        && parameters.jump_log_volatility >= 0.0f;
}

inline const char* parameter_domain_error(const ModelParameters& parameters) {
    if (valid_parameters(parameters)) return nullptr;
    if (!std::isfinite(parameters.spot) || !(parameters.spot > 0.0f))
        return "spot must be finite and positive.";
    if (!std::isfinite(parameters.risk_free_rate))
        return "risk_free_rate must be finite.";
    if (!std::isfinite(parameters.dividend_yield))
        return "dividend_yield must be finite.";
    if (!std::isfinite(parameters.volatility)
        || !(parameters.volatility > 0.0f))
        return "volatility must be finite and positive.";
    if (!std::isfinite(parameters.jump_intensity)
        || parameters.jump_intensity < 0.0f)
        return "jump_intensity must be finite and non-negative.";
    if (!std::isfinite(parameters.jump_log_mean))
        return "jump_log_mean must be finite.";
    if (!std::isfinite(parameters.jump_log_volatility)
        || parameters.jump_log_volatility < 0.0f)
        return "jump_log_volatility must be finite and non-negative.";
    return nullptr;
}

inline void validate_parameters(
    const ModelParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::model::equity::merton
