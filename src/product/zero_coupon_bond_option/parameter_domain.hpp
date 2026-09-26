// Zero-coupon-bond-option parameter-domain predicates.
#pragma once

#include "product/zero_coupon_bond_option/parameters.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace ai_factory::workbench::product::zero_coupon_bond_option {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const ZeroCouponBondOptionParameters& product
) {
    return finite_parameter(product.notional) && product.notional > 0.0f
        && finite_parameter(product.strike) && product.strike > 0.0f
        && product.option_expiry_days > 0U
        && product.bond_maturity_days > product.option_expiry_days;
}

}  // namespace ai_factory::workbench::product::zero_coupon_bond_option
