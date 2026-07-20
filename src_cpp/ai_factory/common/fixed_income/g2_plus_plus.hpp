#pragma once

#include "ai_factory/common/fixed_income/nelson_siegel.hpp"

#include <cmath>

#ifdef __CUDACC__
#define AI_FACTORY_HD __host__ __device__ __forceinline__
#else
#define AI_FACTORY_HD inline
#endif

namespace ai_factory::fixed_income {

AI_FACTORY_HD double ou_b(double mean_reversion, double horizon) {
    return -expm1(-mean_reversion * horizon) / mean_reversion;
}

AI_FACTORY_HD double ou_state_variance(
    double mean_reversion, double volatility, double horizon
) {
    return volatility * volatility
           * (-expm1(-2.0 * mean_reversion * horizon))
           / (2.0 * mean_reversion);
}

AI_FACTORY_HD double ou_integral_variance(
    double mean_reversion, double volatility, double horizon
) {
    const double a = mean_reversion;
    const double one_minus_e = -expm1(-a * horizon);
    const double one_minus_e2 = -expm1(-2.0 * a * horizon);
    return volatility * volatility
           * (horizon - 2.0 * one_minus_e / a
              + one_minus_e2 / (2.0 * a))
           / (a * a);
}

AI_FACTORY_HD double ou_state_integral_covariance(
    double mean_reversion, double volatility, double horizon
) {
    const double one_minus_e = -expm1(-mean_reversion * horizon);
    return volatility * volatility * one_minus_e * one_minus_e
           / (2.0 * mean_reversion * mean_reversion);
}

AI_FACTORY_HD double g2_cross_integral_covariance(
    double a,
    double sigma,
    double b,
    double eta,
    double rho,
    double horizon
) {
    return rho * sigma * eta / (a * b)
           * (horizon - (-expm1(-a * horizon)) / a
              - (-expm1(-b * horizon)) / b
              + (-expm1(-(a + b) * horizon)) / (a + b));
}

AI_FACTORY_HD double g2_integral_variance(
    double a,
    double sigma,
    double b,
    double eta,
    double rho,
    double horizon
) {
    return ou_integral_variance(a, sigma, horizon)
           + ou_integral_variance(b, eta, horizon)
           + 2.0 * g2_cross_integral_covariance(
               a, sigma, b, eta, rho, horizon
           );
}

AI_FACTORY_HD double g2_bond_price(
    double x,
    double y,
    double time,
    double maturity,
    double a,
    double sigma,
    double b,
    double eta,
    double rho,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    const double horizon = maturity - time;
    const double adjustment = 0.5
        * (g2_integral_variance(a, sigma, b, eta, rho, horizon)
           - g2_integral_variance(a, sigma, b, eta, rho, maturity)
           + g2_integral_variance(a, sigma, b, eta, rho, time));
    return nelson_siegel_discount(maturity, beta0, beta1, beta2, tau)
           / nelson_siegel_discount(time, beta0, beta1, beta2, tau)
           * exp(adjustment - ou_b(a, horizon) * x - ou_b(b, horizon) * y);
}

AI_FACTORY_HD double g2_path_discount(
    double integrated_x,
    double integrated_y,
    double time,
    double a,
    double sigma,
    double b,
    double eta,
    double rho,
    double beta0,
    double beta1,
    double beta2,
    double tau
) {
    return nelson_siegel_discount(time, beta0, beta1, beta2, tau)
           * exp(
               -0.5 * g2_integral_variance(
                   a, sigma, b, eta, rho, time
               )
               - integrated_x - integrated_y
           );
}

struct G2Transition {
    double decay_x;
    double decay_y;
    double integral_x_loading;
    double integral_y_loading;
    double cholesky[10];
};

AI_FACTORY_HD G2Transition make_g2_transition(
    double a,
    double sigma,
    double b,
    double eta,
    double rho,
    double horizon
) {
    G2Transition result{};
    result.decay_x = exp(-a * horizon);
    result.decay_y = exp(-b * horizon);
    result.integral_x_loading = ou_b(a, horizon);
    result.integral_y_loading = ou_b(b, horizon);

    double covariance[4][4]{};
    covariance[0][0] = ou_state_variance(a, sigma, horizon);
    covariance[1][1] = ou_state_variance(b, eta, horizon);
    covariance[2][2] = ou_integral_variance(a, sigma, horizon);
    covariance[3][3] = ou_integral_variance(b, eta, horizon);
    covariance[0][1] = covariance[1][0] = rho * sigma * eta
        * (-expm1(-(a + b) * horizon)) / (a + b);
    covariance[0][2] = covariance[2][0] =
        ou_state_integral_covariance(a, sigma, horizon);
    covariance[1][3] = covariance[3][1] =
        ou_state_integral_covariance(b, eta, horizon);
    covariance[0][3] = covariance[3][0] = rho * sigma * eta / b
        * ((-expm1(-a * horizon)) / a
           - (-expm1(-(a + b) * horizon)) / (a + b));
    covariance[1][2] = covariance[2][1] = rho * sigma * eta / a
        * ((-expm1(-b * horizon)) / b
           - (-expm1(-(a + b) * horizon)) / (a + b));
    covariance[2][3] = covariance[3][2] =
        g2_cross_integral_covariance(a, sigma, b, eta, rho, horizon);

    double lower[4][4]{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column <= row; ++column) {
            double value = covariance[row][column];
            for (int inner = 0; inner < column; ++inner) {
                value -= lower[row][inner] * lower[column][inner];
            }
            lower[row][column] = row == column
                ? sqrt(value > 1.0e-30 ? value : 1.0e-30)
                : value / lower[column][column];
        }
    }
    int flat = 0;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column <= row; ++column) {
            result.cholesky[flat++] = lower[row][column];
        }
    }
    return result;
}

AI_FACTORY_HD void apply_g2_transition(
    const G2Transition& transition,
    double z0,
    double z1,
    double z2,
    double z3,
    double& x,
    double& y,
    double& integrated_x,
    double& integrated_y
) {
    const double previous_x = x;
    const double previous_y = y;
    const double noise_x = transition.cholesky[0] * z0;
    const double noise_y = transition.cholesky[1] * z0
                           + transition.cholesky[2] * z1;
    const double noise_ix = transition.cholesky[3] * z0
                            + transition.cholesky[4] * z1
                            + transition.cholesky[5] * z2;
    const double noise_iy = transition.cholesky[6] * z0
                            + transition.cholesky[7] * z1
                            + transition.cholesky[8] * z2
                            + transition.cholesky[9] * z3;
    x = transition.decay_x * previous_x + noise_x;
    y = transition.decay_y * previous_y + noise_y;
    integrated_x += transition.integral_x_loading * previous_x + noise_ix;
    integrated_y += transition.integral_y_loading * previous_y + noise_iy;
}

}  // namespace ai_factory::fixed_income

#undef AI_FACTORY_HD
