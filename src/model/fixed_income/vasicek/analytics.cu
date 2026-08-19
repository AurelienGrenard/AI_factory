// Closed-form fixed-income analytics for the Vasicek model.
#pragma once

#include "model/fixed_income/vasicek/analytics.cuh"

// Include exact Vasicek moments and transitions used by the analytics below.
#include "model/fixed_income/vasicek/dynamics.cu"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::vasicek {

// ==================== Model-specific implementation =======================

namespace {

constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

// Evaluate the standard normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    return 0.5f * erfcf(-value * kInverseSqrtTwo);
}

struct AffineBondCoefficients {
    float log_A;
    float B;
};

// Compute log(A) and B together from one shared set of integral moments.
__device__ __forceinline__ AffineBondCoefficients affine_bond_coefficients(
    const VasicekProcessParameters& parameters,
    float delta
) {
    const VasicekIntegralMoments moments = integral_moments(
        parameters, delta
    );
    return {
        0.5f * moments.variance - moments.mean_increment,
        moments.state_loading,
    };
}

// Price a call (+1) or put (-1) with one shared Black-style expression.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const VasicekModelParameters& parameters,
    float state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    const float time_to_expiry = option_expiry - valuation_time;
    const VasicekIntegralMoments expiry_moments = integral_moments(
        parameters.process, time_to_expiry
    );
    const float expiry_log_bond = fmaf(
        -expiry_moments.state_loading,
        state,
        0.5f * expiry_moments.variance - expiry_moments.mean_increment
    );
    const float underlying_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time, bond_maturity
    );
    const float expiry_bond = expf(expiry_log_bond);
    const float underlying_bond = expf(underlying_log_bond);

    // Recover Var[x_expiry|F_t] from the already computed Vasicek loading.
    const float a = parameters.process.mean_reversion;
    const float one_minus_decay = a * expiry_moments.state_loading;
    const float expiry_state_variance =
        parameters.process.volatility * parameters.process.volatility
        * expiry_moments.state_loading * (1.0f - 0.5f * one_minus_decay);
    const float bond_loading = integral_state_loading(
        a, bond_maturity - option_expiry
    );
    const float total_volatility =
        bond_loading * sqrtf(fmaxf(expiry_state_variance, 0.0f));
    if (total_volatility <= 1.0e-7f) {
        return fmaxf(
            option_sign * (underlying_bond - strike * expiry_bond),
            0.0f
        );
    }

    const float d1 =
        (underlying_log_bond - expiry_log_bond - logf(strike))
            / total_volatility
        + 0.5f * total_volatility;
    const float d2 = d1 - total_volatility;
    return fmaxf(
        option_sign
            * (
                underlying_bond * normal_cdf(option_sign * d1)
                - strike * expiry_bond
                    * normal_cdf(option_sign * d2)
            ),
        0.0f
    );
}

}  // namespace

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the multiplicative affine prefactor.
__device__ __forceinline__ float log_A(
    const VasicekModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    ).log_A;
}

// Exponentiate the affine prefactor only for callers requesting A itself.
__device__ __forceinline__ float A(
    const VasicekModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return expf(log_A(parameters, valuation_time, maturity));
}

// Return the current-rate loading in the affine bond expression.
__device__ __forceinline__ float B(
    const VasicekModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return integral_state_loading(
        parameters.process.mean_reversion, maturity - valuation_time
    );
}

// Evaluate log(A)-B*r from one grouped coefficient calculation.
__device__ __forceinline__ float log_zero_coupon_bond(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    const AffineBondCoefficients coefficients = affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    );
    return fmaf(-coefficients.B, state, coefficients.log_A);
}

// The Vasicek state is the short rate, so its integral is the log discount.
__device__ __forceinline__ float log_discount_factor(
    float state_integral
) {
    return -state_integral;
}

// Exponentiate the accumulated short-rate integral only when required.
__device__ __forceinline__ float discount_factor(
    float state_integral
) {
    return expf(log_discount_factor(state_integral));
}

// Price one zero-coupon from the conditional Gaussian rate integral.
__device__ __forceinline__ float zero_coupon_bond(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return expf(log_zero_coupon_bond(
        parameters, state, valuation_time, maturity
    ));
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    return zero_coupon_bond_option_price(
        parameters,
        state,
        1.0f,
        valuation_time,
        option_expiry,
        bond_maturity,
        strike
    );
}

// Apply the closed-form put formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    return zero_coupon_bond_option_price(
        parameters,
        state,
        -1.0f,
        valuation_time,
        option_expiry,
        bond_maturity,
        strike
    );
}

// Build one simple forward rate from two conditional zero-coupons.
__device__ __forceinline__ float forward_rate(
    const VasicekModelParameters& parameters,
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
    const VasicekModelParameters& parameters,
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
        annuity = fmaf(
            accrual_periods[payment],
            current_bond,
            annuity
        );
        end_bond = current_bond;
    }
    const float start_bond = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    // Value the floating leg from its start and final zero-coupon bonds.
    return (start_bond - end_bond) / annuity;
}

}  // namespace ai_factory::workbench::model::vasicek
