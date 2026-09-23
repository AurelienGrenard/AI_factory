// CIR parameter-domain predicate and host validation diagnostics for standalone or shifted factors.
#pragma once

#include "model/fixed_income/cir/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::fixed_income::cir {

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
    return finite_parameter(parameters.process.mean_reversion)
        && parameters.process.mean_reversion > 0.0f
        && finite_parameter(parameters.process.long_term_mean)
        && parameters.process.long_term_mean > 0.0f
        && finite_parameter(parameters.process.volatility)
        && parameters.process.volatility > 0.0f
        && finite_parameter(parameters.initial_state)
        && parameters.initial_state >= 0.0f;
}

inline const char* parameter_domain_error(const ModelParameters& parameters) {
    if (valid_parameters(parameters)) return nullptr;
    if (!std::isfinite(parameters.process.mean_reversion)
        || !(parameters.process.mean_reversion > 0.0f))
        return "mean_reversion must be finite and positive.";
    if (!std::isfinite(parameters.process.long_term_mean)
        || !(parameters.process.long_term_mean > 0.0f))
        return "long_term_mean must be finite and positive.";
    if (!std::isfinite(parameters.process.volatility)
        || !(parameters.process.volatility > 0.0f))
        return "volatility must be finite and positive.";
    if (!std::isfinite(parameters.initial_state)
        || !(parameters.initial_state >= 0.0f))
        return "initial_state must be finite and non-negative.";
    return nullptr;
}

inline void validate_parameters(
    const ModelParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::model::fixed_income::cir
