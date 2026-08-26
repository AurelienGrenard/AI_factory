// Discounted lognormal option primitives shared across asset classes.
#pragma once

#include "common/normal_distribution.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench {

__device__ __forceinline__ float fp32_infinity() {
    return __int_as_float(0x7f800000);
}

struct DiscountedLognormalOptionContext {
    float underlying_log_level;
    float strike_discount_log_level;
    float strike_discount_level;
};

struct LognormalOptionDValues {
    float d1;
    float d2;
};

struct DiscountedLognormalOptionValues {
    float discounted_underlying;
    float discounted_strike;
    float d1;
    float d2;
};

__device__ __forceinline__ LognormalOptionDValues lognormal_option_d_values(
    const DiscountedLognormalOptionContext& context,
    float total_volatility,
    float strike
) {
    const float d1 =
        (context.underlying_log_level
         - context.strike_discount_log_level
         - logf(strike))
            / total_volatility
        + 0.5f * total_volatility;
    return {d1, d1 - total_volatility};
}

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_discounted_lognormal_option_values(
    const DiscountedLognormalOptionContext& context,
    float total_volatility,
    float strike
) {
    const float discounted_underlying = expf(
        context.underlying_log_level
    );
    const float discounted_strike =
        strike * context.strike_discount_level;
    if (total_volatility <= 1.0e-7f) {
        const float log_moneyness =
            context.underlying_log_level
            - context.strike_discount_log_level
            - logf(strike);
        const float limiting_d = log_moneyness > 0.0f
            ? fp32_infinity()
            : (log_moneyness < 0.0f
                ? -fp32_infinity()
                : 0.0f);
        return {
            discounted_underlying,
            discounted_strike,
            limiting_d,
            limiting_d,
        };
    }
    const LognormalOptionDValues values = lognormal_option_d_values(
        context, total_volatility, strike
    );
    return {
        discounted_underlying,
        discounted_strike,
        values.d1,
        values.d2,
    };
}

__device__ __forceinline__ float discounted_lognormal_option_price(
    const DiscountedLognormalOptionValues& values,
    float option_sign
) {
    return fmaxf(
        option_sign
            * (
                values.discounted_underlying
                    * normal_cdf(option_sign * values.d1)
                - values.discounted_strike
                    * normal_cdf(option_sign * values.d2)
            ),
        0.0f
    );
}

// Price call (+1) or put (-1) from discounted levels and total volatility.
__device__ __forceinline__ float discounted_lognormal_option_price(
    const DiscountedLognormalOptionContext& context,
    float total_volatility,
    float strike,
    float option_sign
) {
    const DiscountedLognormalOptionValues values =
        prepare_discounted_lognormal_option_values(
        context, total_volatility, strike
    );
    return discounted_lognormal_option_price(values, option_sign);
}

}  // namespace ai_factory::workbench
