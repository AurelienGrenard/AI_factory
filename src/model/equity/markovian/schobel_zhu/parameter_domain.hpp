// Schobel--Zhu host/device parameter-domain contract.
#pragma once

#include "model/equity/markovian/schobel_zhu/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::schobel_zhu {

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
        && finite_parameter(parameters.initial_volatility)
        && finite_parameter(parameters.mean_reversion)
        && parameters.mean_reversion > 0.0f
        && finite_parameter(parameters.long_run_volatility)
        && finite_parameter(parameters.volatility_of_volatility)
        && parameters.volatility_of_volatility > 0.0f
        && finite_parameter(parameters.correlation)
        && parameters.correlation > -1.0f
        && parameters.correlation < 1.0f;
}

inline const char* parameter_domain_error(const ModelParameters& parameters) {
    if (valid_parameters(parameters)) return nullptr;
    if (!std::isfinite(parameters.spot) || !(parameters.spot > 0.0f))
        return "spot must be finite and positive.";
    if (!std::isfinite(parameters.risk_free_rate))
        return "risk_free_rate must be finite.";
    if (!std::isfinite(parameters.dividend_yield))
        return "dividend_yield must be finite.";
    if (!std::isfinite(parameters.initial_volatility))
        return "initial_volatility must be finite.";
    if (!std::isfinite(parameters.mean_reversion)
        || !(parameters.mean_reversion > 0.0f))
        return "mean_reversion must be finite and positive.";
    if (!std::isfinite(parameters.long_run_volatility))
        return "long_run_volatility must be finite.";
    if (!std::isfinite(parameters.volatility_of_volatility)
        || !(parameters.volatility_of_volatility > 0.0f))
        return "volatility_of_volatility must be finite and positive.";
    if (!std::isfinite(parameters.correlation)
        || !(parameters.correlation > -1.0f
             && parameters.correlation < 1.0f))
        return "correlation must lie strictly between -1 and 1.";
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

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
