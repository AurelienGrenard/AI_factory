// Variance-Gamma host/device parameter-domain contract and diagnostics.
#pragma once

#include "model/equity/markovian/variance_gamma/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::model::equity::variance_gamma {

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
    if (!(finite_parameter(parameters.spot) && parameters.spot > 0.0f)
        || !finite_parameter(parameters.risk_free_rate)
        || !finite_parameter(parameters.dividend_yield)
        || !(finite_parameter(parameters.sigma) && parameters.sigma > 0.0f)
        || !(finite_parameter(parameters.nu) && parameters.nu > 0.0f)
        || !finite_parameter(parameters.theta)) {
        return false;
    }
    const float sigma2_nu =
        parameters.sigma * parameters.sigma * parameters.nu;
    const float first_moment = 1.0f
        - parameters.theta * parameters.nu - 0.5f * sigma2_nu;
    const float second_moment = 1.0f
        - 2.0f * parameters.theta * parameters.nu - 2.0f * sigma2_nu;
    return finite_parameter(first_moment) && first_moment > 0.0f
        && finite_parameter(second_moment) && second_moment > 0.0f;
}

inline const char* parameter_domain_error(const ModelParameters& parameters) {
    if (valid_parameters(parameters)) return nullptr;
    if (!std::isfinite(parameters.spot) || !(parameters.spot > 0.0f))
        return "spot must be finite and positive.";
    if (!std::isfinite(parameters.risk_free_rate))
        return "risk_free_rate must be finite.";
    if (!std::isfinite(parameters.dividend_yield))
        return "dividend_yield must be finite.";
    if (!std::isfinite(parameters.sigma) || !(parameters.sigma > 0.0f))
        return "sigma must be finite and positive.";
    if (!std::isfinite(parameters.nu) || !(parameters.nu > 0.0f))
        return "nu must be finite and positive.";
    if (!std::isfinite(parameters.theta))
        return "theta must be finite.";
    const float first_moment = 1.0f - parameters.theta * parameters.nu
        - 0.5f * parameters.sigma * parameters.sigma * parameters.nu;
    if (!std::isfinite(first_moment) || !(first_moment > 0.0f))
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

}  // namespace ai_factory::workbench::model::equity::variance_gamma
