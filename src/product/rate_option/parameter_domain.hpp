// Rate-option parameter-domain predicates shared by datasets and sensitivities.
#pragma once

#include "product/rate_option/parameters.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace ai_factory::workbench::product::rate_option {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const RateOptionParameters& product
) {
    if (!finite_parameter(product.notional) || !(product.notional > 0.0f)
        || !finite_parameter(product.strike)
        || product.fixing_time_days == 0U
        || !(product.payment_time_days > product.fixing_time_days)
        || product.accrual_period_days == 0U
        || product.payment_time_days - product.fixing_time_days
            != product.accrual_period_days) {
        return false;
    }
    const float accrual_years =
        static_cast<float>(product.accrual_period_days) / 252.0f;
    return fmaf(accrual_years, product.strike, 1.0f) > 0.0f;
}

}  // namespace ai_factory::workbench::product::rate_option
