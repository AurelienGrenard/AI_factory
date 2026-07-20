#pragma once

#include "ai_factory/common/fixed_income/cir.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"

#include <cmath>

#ifdef __CUDACC__
#define AI_FACTORY_HD __host__ __device__ __forceinline__
#else
#define AI_FACTORY_HD inline
#endif

namespace ai_factory::fixed_income {

AI_FACTORY_HD double cir_plus_plus_base_discount(
    double maturity,
    double initial_factor,
    double kappa,
    double theta,
    double volatility
) {
    return cir_bond_price(
        initial_factor, kappa, theta, volatility, maturity
    );
}

AI_FACTORY_HD double cir_plus_plus_bond_price(
    double factor,
    double time,
    double maturity,
    double initial_factor,
    double kappa,
    double theta,
    double volatility,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    const double market_ratio = nelson_siegel_discount(
        maturity, beta0, beta1, beta2, tau
    ) / nelson_siegel_discount(time, beta0, beta1, beta2, tau);
    const double base_ratio = cir_plus_plus_base_discount(
        time, initial_factor, kappa, theta, volatility
    ) / cir_plus_plus_base_discount(
        maturity, initial_factor, kappa, theta, volatility
    );
    return market_ratio * base_ratio
           * cir_bond_price(
               factor, kappa, theta, volatility, maturity - time
           );
}

AI_FACTORY_HD double cir_plus_plus_bond_a(
    double time,
    double maturity,
    double initial_factor,
    double kappa,
    double theta,
    double volatility,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    const double market_ratio = nelson_siegel_discount(
        maturity, beta0, beta1, beta2, tau
    ) / nelson_siegel_discount(time, beta0, beta1, beta2, tau);
    const double base_ratio = cir_plus_plus_base_discount(
        time, initial_factor, kappa, theta, volatility
    ) / cir_plus_plus_base_discount(
        maturity, initial_factor, kappa, theta, volatility
    );
    return market_ratio * base_ratio
           * cir_bond_a(kappa, theta, volatility, maturity - time);
}

AI_FACTORY_HD double cir_plus_plus_path_discount(
    double integrated_factor,
    double time,
    double initial_factor,
    double kappa,
    double theta,
    double volatility,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    return nelson_siegel_discount(time, beta0, beta1, beta2, tau)
           / cir_plus_plus_base_discount(
               time, initial_factor, kappa, theta, volatility
           )
           * exp(-integrated_factor);
}

}  // namespace ai_factory::fixed_income

#undef AI_FACTORY_HD
