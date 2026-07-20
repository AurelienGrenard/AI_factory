#pragma once

#include <cmath>

#ifdef __CUDACC__
#define AI_FACTORY_HD __host__ __device__ __forceinline__
#else
#define AI_FACTORY_HD inline
#endif

namespace ai_factory::fixed_income {

AI_FACTORY_HD double normal_cdf(double value) {
    return 0.5 * erfc(-value * 0.70710678118654752440);
}

AI_FACTORY_HD double shifted_black_option(
    double forward,
    double strike,
    double displacement,
    double total_volatility,
    int direction
) {
    const double shifted_forward = forward + displacement;
    const double shifted_strike = strike + displacement;
    if (total_volatility <= 0.0) {
        return fmax(
            static_cast<double>(direction) * (forward - strike), 0.0
        );
    }
    const double d1 = log(shifted_forward / shifted_strike)
        / total_volatility + 0.5 * total_volatility;
    const double d2 = d1 - total_volatility;
    const double sign = static_cast<double>(direction);
    return sign * (
        shifted_forward * normal_cdf(sign * d1)
        - shifted_strike * normal_cdf(sign * d2)
    );
}

}  // namespace ai_factory::fixed_income

#undef AI_FACTORY_HD
