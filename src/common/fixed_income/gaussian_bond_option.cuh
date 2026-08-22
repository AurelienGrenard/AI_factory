// Discounted lognormal bond-option expression shared by Gaussian models.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench::fixed_income {

struct GaussianBondOptionDiscountContext {
    float expiry_log_bond;
    float expiry_bond;
};

constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

__device__ __forceinline__ float normal_cdf(float value) {
    return 0.5f * erfcf(-value * kInverseSqrtTwo);
}

// Price a call (+1) or put (-1) from discounted bond levels and total sigma.
__device__ __forceinline__ float discounted_lognormal_bond_option_price(
    const GaussianBondOptionDiscountContext& context,
    float underlying_log_bond,
    float total_volatility,
    float strike,
    float option_sign
) {
    const float underlying_bond = expf(underlying_log_bond);
    if (total_volatility <= 1.0e-7f) {
        return fmaxf(
            option_sign
                * (underlying_bond - strike * context.expiry_bond),
            0.0f
        );
    }

    const float d1 =
        (underlying_log_bond - context.expiry_log_bond - logf(strike))
            / total_volatility
        + 0.5f * total_volatility;
    const float d2 = d1 - total_volatility;
    return fmaxf(
        option_sign
            * (
                underlying_bond * normal_cdf(option_sign * d1)
                - strike * context.expiry_bond
                    * normal_cdf(option_sign * d2)
            ),
        0.0f
    );
}

}  // namespace ai_factory::workbench::fixed_income
