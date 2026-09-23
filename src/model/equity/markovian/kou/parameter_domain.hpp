// Kou host/device parameter-domain contract and diagnostics.
#pragma once

#include "model/equity/markovian/kou/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::kou {

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
        && finite_parameter(parameters.up_probability)
        && parameters.up_probability > 0.0f
        && parameters.up_probability < 1.0f
        && finite_parameter(parameters.positive_jump_rate)
        && parameters.positive_jump_rate > 2.0f
        && finite_parameter(parameters.negative_jump_rate)
        && parameters.negative_jump_rate > 0.0f;
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
    if (!std::isfinite(parameters.up_probability)
        || !(parameters.up_probability > 0.0f
             && parameters.up_probability < 1.0f))
        return "up_probability must lie strictly between zero and one.";
    if (!std::isfinite(parameters.positive_jump_rate)
        || !(parameters.positive_jump_rate > 2.0f))
        return "positive_jump_rate must exceed two for finite Monte Carlo "
               "payoff variance.";
    if (!std::isfinite(parameters.negative_jump_rate)
        || !(parameters.negative_jump_rate > 0.0f))
        return "negative_jump_rate must be finite and positive.";
    return nullptr;
}

inline void validate_parameters(
    const ModelParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters)) {
        throw std::invalid_argument(prefix + error);
    }
}

}  // namespace ai_factory::workbench::model::equity::kou
