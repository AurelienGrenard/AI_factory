// Digital-option host/device parameter domain.
#pragma once

#include "product/digital_option/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::digital_option {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const DigitalOptionParameters& product
) {
    return finite_parameter(product.strike) && product.strike > 0.0f
        && product.maturity_days > 0U
        && finite_parameter(product.cash_payoff) && product.cash_payoff > 0.0f;
}

inline void validate_parameters(
    const DigitalOptionParameters& product,
    const std::string& prefix
) {
    if (!std::isfinite(product.strike) || !(product.strike > 0.0f)) {
        throw std::invalid_argument(
            prefix + "strike must be finite and positive."
        );
    }
    if (product.maturity_days == 0U) {
        throw std::invalid_argument(
            prefix + "maturity must be a positive business-day count."
        );
    }
    if (!std::isfinite(product.cash_payoff)
        || !(product.cash_payoff > 0.0f)) {
        throw std::invalid_argument(
            prefix + "cash_payoff must be finite and positive."
        );
    }
}

}  // namespace ai_factory::workbench::product::digital_option
