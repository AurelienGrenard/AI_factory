#pragma once

#include "ai_factory/common/fixed_income/nelson_siegel.hpp"

#include <cmath>

#ifdef __CUDACC__
#define AI_FACTORY_HD __host__ __device__ __forceinline__
#else
#define AI_FACTORY_HD inline
#endif

namespace ai_factory::fixed_income {

AI_FACTORY_HD double hull_white_b(double mean_reversion, double horizon) {
    return -expm1(-mean_reversion * horizon) / mean_reversion;
}

AI_FACTORY_HD double hull_white_state_variance(
    double mean_reversion,
    double volatility,
    double horizon
) {
    return volatility * volatility
           * (-expm1(-2.0 * mean_reversion * horizon))
           / (2.0 * mean_reversion);
}

AI_FACTORY_HD double hull_white_integral_variance(
    double mean_reversion,
    double volatility,
    double horizon
) {
    const double a = mean_reversion;
    const double one_minus_e = -expm1(-a * horizon);
    const double one_minus_e2 = -expm1(-2.0 * a * horizon);
    const double bracket =
        horizon - 2.0 * one_minus_e / a + one_minus_e2 / (2.0 * a);
    return volatility * volatility * bracket / (a * a);
}

AI_FACTORY_HD double hull_white_state_integral_covariance(
    double mean_reversion,
    double volatility,
    double horizon
) {
    const double one_minus_e = -expm1(-mean_reversion * horizon);
    return volatility * volatility * one_minus_e * one_minus_e
           / (2.0 * mean_reversion * mean_reversion);
}

AI_FACTORY_HD double hull_white_deterministic_integral(
    double horizon,
    double mean_reversion,
    double volatility,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    const double discount = nelson_siegel_discount(
        horizon, beta0, beta1, beta2, tau
    );
    return -log(discount)
           + 0.5 * hull_white_integral_variance(
               mean_reversion, volatility, horizon
           );
}

AI_FACTORY_HD double hull_white_bond_a(
    double time,
    double maturity,
    double mean_reversion,
    double volatility,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    const double p0_time = nelson_siegel_discount(
        time, beta0, beta1, beta2, tau
    );
    const double p0_maturity = nelson_siegel_discount(
        maturity, beta0, beta1, beta2, tau
    );
    const double variance_adjustment =
        hull_white_integral_variance(mean_reversion, volatility, maturity)
        - hull_white_integral_variance(mean_reversion, volatility, time)
        - hull_white_integral_variance(
            mean_reversion, volatility, maturity - time
        );
    return (p0_maturity / p0_time) * exp(-0.5 * variance_adjustment);
}

AI_FACTORY_HD double hull_white_bond_price(
    double state,
    double time,
    double maturity,
    double mean_reversion,
    double volatility,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    return hull_white_bond_a(
               time,
               maturity,
               mean_reversion,
               volatility,
               beta0,
               beta1,
               beta2,
               tau
           )
           * exp(-hull_white_b(mean_reversion, maturity - time) * state);
}

}  // namespace ai_factory::fixed_income

#undef AI_FACTORY_HD
