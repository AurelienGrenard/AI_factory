// Discounted lognormal bond-option expression shared by Gaussian models.
#pragma once

#include "common/lognormal_option.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::fixed_income {

struct GaussianBondOptionDiscountContext {
    float expiry_log_bond;
    float expiry_bond;
};

// Price a call (+1) or put (-1) from discounted bond levels and total sigma.
__device__ __forceinline__ float discounted_lognormal_bond_option_price(
    const GaussianBondOptionDiscountContext& context,
    float underlying_log_bond,
    float total_volatility,
    float strike,
    float option_sign
) {
    return ::ai_factory::workbench::discounted_lognormal_option_price(
        {
            underlying_log_bond,
            context.expiry_log_bond,
            context.expiry_bond,
        },
        total_volatility,
        strike,
        option_sign
    );
}

}  // namespace ai_factory::workbench::fixed_income
