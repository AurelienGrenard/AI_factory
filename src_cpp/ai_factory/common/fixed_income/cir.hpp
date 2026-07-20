#pragma once

#include <cmath>

#ifdef __CUDACC__
#define AI_FACTORY_HD __host__ __device__ __forceinline__
#else
#define AI_FACTORY_HD inline
#endif

namespace ai_factory::fixed_income {

AI_FACTORY_HD double cir_bond_b(double kappa, double volatility, double horizon) {
    const double gamma = sqrt(kappa * kappa + 2.0 * volatility * volatility);
    const double e = expm1(gamma * horizon);
    return 2.0 * e / ((gamma + kappa) * e + 2.0 * gamma);
}

AI_FACTORY_HD double cir_bond_a(
    double kappa, double theta, double volatility, double horizon
) {
    const double gamma = sqrt(kappa * kappa + 2.0 * volatility * volatility);
    const double e = expm1(gamma * horizon);
    const double denominator = (gamma + kappa) * e + 2.0 * gamma;
    const double log_base = log(2.0 * gamma / denominator)
                            + 0.5 * (kappa + gamma) * horizon;
    return exp(2.0 * kappa * theta / (volatility * volatility) * log_base);
}

AI_FACTORY_HD double cir_bond_price(
    double short_rate,
    double kappa,
    double theta,
    double volatility,
    double horizon
) {
    return cir_bond_a(kappa, theta, volatility, horizon)
           * exp(-cir_bond_b(kappa, volatility, horizon) * short_rate);
}

}  // namespace ai_factory::fixed_income

#undef AI_FACTORY_HD
