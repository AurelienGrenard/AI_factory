// Asset-or-nothing host/device parameter domain.
#pragma once

#include "product/asset_or_nothing_option/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::asset_or_nothing_option {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const AssetOrNothingOptionParameters& product
) {
    return finite_parameter(product.strike) && product.strike > 0.0f
        && product.maturity_days > 0U;
}

inline void validate_parameters(
    const AssetOrNothingOptionParameters& product,
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
}

}  // namespace ai_factory::workbench::product::asset_or_nothing_option
