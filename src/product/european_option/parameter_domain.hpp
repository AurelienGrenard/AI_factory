// European-option host/device domain predicate and host validation diagnostics.
#pragma once

#include "product/european_option/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::european_option {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const EuropeanOptionParameters& parameters
) {
    return finite_parameter(parameters.strike) && parameters.strike > 0.0f
        && parameters.maturity_days != 0U;
}

inline const char* parameter_domain_error(
    const EuropeanOptionParameters& parameters
) {
    if (valid_parameters(parameters)) return nullptr;
    if (!std::isfinite(parameters.strike) || !(parameters.strike > 0.0f))
        return "strike must be finite and positive.";
    if (parameters.maturity_days == 0U)
        return "maturity must be a positive business-day count.";
    return nullptr;
}

inline void validate_parameters(
    const EuropeanOptionParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::product::european_option
