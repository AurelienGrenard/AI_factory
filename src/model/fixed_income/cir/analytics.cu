// Closed-form affine fixed-income analytics for the CIR model.
#pragma once

#include "model/fixed_income/cir/analytics.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::cir {

// ==================== Model-specific implementation =======================

namespace {

struct AffineBondCoefficients {
    float log_A;
    float B;
};

// Compute log(A) and B together while sharing gamma and exp(-gamma*tau).
__device__ __forceinline__ AffineBondCoefficients affine_bond_coefficients(
    const CirProcessParameters& process,
    float time_to_maturity
) {
    const float kappa = process.mean_reversion;
    const float sigma_squared = process.volatility * process.volatility;
    const float gamma = sqrtf(kappa * kappa + 2.0f * sigma_squared);
    const float one_minus_gamma_decay = -expm1f(
        -gamma * time_to_maturity
    );
    const float denominator = fmaf(
        kappa - gamma,
        one_minus_gamma_decay,
        2.0f * gamma
    );
    const float relative_denominator_increment =
        (kappa - gamma) * one_minus_gamma_decay / (2.0f * gamma);
    const float log_base = -log1pf(relative_denominator_increment)
        + 0.5f * (kappa - gamma) * time_to_maturity;
    return {
        2.0f * kappa * process.long_term_mean / sigma_squared * log_base,
        2.0f * one_minus_gamma_decay / denominator,
    };
}

}  // namespace

// ===================== Common fixed-income analytics ======================

// Expose log(A) because bond logarithms should not round-trip through exp/log.
__device__ __forceinline__ float log_A(
    const CirModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    ).log_A;
}

// Expose the textbook multiplicative prefactor used in P=A*exp(-B*r).
__device__ __forceinline__ float A(
    const CirModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return expf(log_A(parameters, valuation_time, maturity));
}

// Expose the textbook state loading used in P=A*exp(-B*r).
__device__ __forceinline__ float B(
    const CirModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    ).B;
}

// Evaluate the affine bond in log space for numerical stability.
__device__ __forceinline__ float log_zero_coupon_bond(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    const AffineBondCoefficients coefficients = affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    );
    return fmaf(
        -coefficients.B,
        state,
        coefficients.log_A
    );
}

// Exponentiate the conditional affine log bond only at the public boundary.
__device__ __forceinline__ float zero_coupon_bond(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return expf(log_zero_coupon_bond(
        parameters, state, valuation_time, maturity
    ));
}

// Build one simple forward rate from two conditional zero-coupons.
__device__ __forceinline__ float forward_rate(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
) {
    const float start_bond = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    const float end_bond = zero_coupon_bond(
        parameters, state, valuation_time, end_time
    );
    return (start_bond / end_bond - 1.0f) / accrual_period;
}

// Divide the conditional floating-leg value by the fixed-leg annuity.
__device__ __forceinline__ float swap_rate(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
) {
    float annuity = 0.0f;
    float end_bond = 0.0f;
    for (std::size_t payment = 0U;
         payment < payment_count;
         ++payment) {
        const float current_bond = zero_coupon_bond(
            parameters,
            state,
            valuation_time,
            payment_times[payment]
        );
        annuity = fmaf(accrual_periods[payment], current_bond, annuity);
        end_bond = current_bond;
    }
    const float start_bond = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    return (start_bond - end_bond) / annuity;
}

}  // namespace ai_factory::workbench::model::cir
