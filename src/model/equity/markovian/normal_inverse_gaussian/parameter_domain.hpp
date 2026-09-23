// NIG host/device parameter-domain contract and diagnostics.
#pragma once

#include "model/equity/markovian/normal_inverse_gaussian/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline float absolute_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::fabsf(value);
#else
    return std::fabs(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const ModelParameters& parameters
) {
    if (!(finite_parameter(parameters.spot) && parameters.spot > 0.0f)
        || !finite_parameter(parameters.risk_free_rate)
        || !finite_parameter(parameters.dividend_yield)
        || !(finite_parameter(parameters.alpha) && parameters.alpha > 0.0f)
        || !finite_parameter(parameters.beta)
        || !(finite_parameter(parameters.delta) && parameters.delta > 0.0f)) {
        return false;
    }
    return parameters.alpha > absolute_parameter(parameters.beta)
        && parameters.alpha > absolute_parameter(parameters.beta + 1.0f)
        && parameters.alpha > absolute_parameter(parameters.beta + 2.0f);
}

inline const char* parameter_domain_error(const ModelParameters& parameters) {
    if (valid_parameters(parameters)) return nullptr;
    if (!std::isfinite(parameters.spot) || !(parameters.spot > 0.0f))
        return "spot must be finite and positive.";
    if (!std::isfinite(parameters.risk_free_rate))
        return "risk_free_rate must be finite.";
    if (!std::isfinite(parameters.dividend_yield))
        return "dividend_yield must be finite.";
    if (!std::isfinite(parameters.alpha) || !(parameters.alpha > 0.0f))
        return "alpha must be finite and positive.";
    if (!std::isfinite(parameters.beta))
        return "beta must be finite.";
    if (!std::isfinite(parameters.delta) || !(parameters.delta > 0.0f))
        return "delta must be finite and positive.";
    if (!(parameters.alpha > std::fabs(parameters.beta)))
        return "alpha must exceed abs(beta).";
    if (!(parameters.alpha > std::fabs(parameters.beta + 1.0f)))
        return "the first exponential moment must be finite.";
    return "the second exponential moment must be finite.";
}

inline void validate_parameters(
    const ModelParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
