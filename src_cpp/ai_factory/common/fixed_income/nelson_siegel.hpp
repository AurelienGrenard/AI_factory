#pragma once

#include <cmath>

#ifdef __CUDACC__
#define AI_FACTORY_HD __host__ __device__ __forceinline__
#else
#define AI_FACTORY_HD inline
#endif

namespace ai_factory::fixed_income {

AI_FACTORY_HD double nelson_siegel_loading(double maturity, double tau) {
    if (maturity == 0.0) {
        return 1.0;
    }
    const double x = maturity / tau;
    return -expm1(-x) / x;
}

AI_FACTORY_HD double nelson_siegel_zero_rate(
    double maturity,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    if (maturity == 0.0) {
        return beta0 + beta1;
    }
    const double x = maturity / tau;
    const double exponential = exp(-x);
    const double loading = -expm1(-x) / x;
    return beta0 + beta1 * loading + beta2 * (loading - exponential);
}

AI_FACTORY_HD double nelson_siegel_discount(
    double maturity,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    if (maturity == 0.0) {
        return 1.0;
    }
    return exp(
        -maturity
        * nelson_siegel_zero_rate(maturity, beta0, beta1, beta2, tau)
    );
}

AI_FACTORY_HD double nelson_siegel_forward(
    double maturity,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    const double x = maturity / tau;
    const double exponential = exp(-x);
    return beta0 + beta1 * exponential + beta2 * x * exponential;
}

}  // namespace ai_factory::fixed_income

#undef AI_FACTORY_HD
