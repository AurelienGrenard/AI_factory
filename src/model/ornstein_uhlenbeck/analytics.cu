// Analytical loading and integral variance of the OU process.
#pragma once

#include "model/ornstein_uhlenbeck/analytics.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {
namespace {

constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

// Evaluate the standard normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    return 0.5f * erfcf(-value * kInverseSqrtTwo);
}

// Return the total log-volatility of a zero-coupon at option expiry.
__device__ __forceinline__ float zero_coupon_bond_option_volatility(
    const OrnsteinUhlenbeckModelParameters& parameters,
    float valuation_time,
    float option_expiry,
    float bond_maturity
) {
    const float a = parameters.dynamics.mean_reversion;
    const float time_to_expiry = option_expiry - valuation_time;
    const float factor_variance =
        parameters.dynamics.volatility * parameters.dynamics.volatility
        * (-expm1f(-2.0f * a * time_to_expiry)) / (2.0f * a);
    const float bond_loading = integral_factor_loading(
        a, bond_maturity - option_expiry
    );
    return bond_loading * sqrtf(fmaxf(factor_variance, 0.0f));
}

// Price a call (+1) or put (-1) with one shared Black-style expression.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    const float expiry_discount = zero_coupon_bond(
        parameters, state, valuation_time, option_expiry
    );
    const float bond_discount = zero_coupon_bond(
        parameters, state, valuation_time, bond_maturity
    );
    const float volatility = zero_coupon_bond_option_volatility(
        parameters, valuation_time, option_expiry, bond_maturity
    );
    if (volatility <= 1.0e-7f) {
        return fmaxf(
            option_sign * (bond_discount - strike * expiry_discount),
            0.0f
        );
    }

    const float d1 =
        logf(bond_discount / (strike * expiry_discount)) / volatility
        + 0.5f * volatility;
    const float d2 = d1 - volatility;
    return fmaxf(
        option_sign
            * (
                bond_discount * normal_cdf(option_sign * d1)
                - strike * expiry_discount
                    * normal_cdf(option_sign * d2)
            ),
        0.0f
    );
}

}  // namespace

// Evaluate the exact loading of the current factor in its future integral.
__device__ __forceinline__ float integral_factor_loading(
    float mean_reversion,
    float delta
) {
    return -expm1f(-mean_reversion * delta) / mean_reversion;
}

// Evaluate the integral variance stably near zero mean reversion.
__device__ __forceinline__ float integral_variance(
    const OrnsteinUhlenbeckDynamicsParameters& parameters,
    float delta
) {
    const float a = parameters.mean_reversion;
    const float scaled_time = a * delta;
    if (fabsf(scaled_time) < 0.02f) {
        const float scaled_time2 = scaled_time * scaled_time;
        const float normalized =
            1.0f / 3.0f
            - scaled_time / 4.0f
            + 7.0f * scaled_time2 / 60.0f
            - scaled_time2 * scaled_time / 24.0f;
        return parameters.volatility * parameters.volatility
            * delta * delta * delta * normalized;
    }

    const float one_minus_decay = -expm1f(-a * delta);
    const float one_minus_decay_squared = -expm1f(-2.0f * a * delta);
    const float bracket =
        delta
        - 2.0f * one_minus_decay / a
        + one_minus_decay_squared / (2.0f * a);
    return parameters.volatility * parameters.volatility
        * bracket / (a * a);
}

// The standalone OU state stores the current short rate directly.
__device__ __forceinline__ float short_rate(
    const OrnsteinUhlenbeckModelParameters&,
    const OrnsteinUhlenbeckState& state,
    float
) {
    return state.factor;
}

// The OU factor is the short rate, so its stored integral is the log discount.
__device__ __forceinline__ float log_discount(
    const OrnsteinUhlenbeckModelParameters&,
    const OrnsteinUhlenbeckState& state,
    float
) {
    return -state.integrated_factor;
}

// Exponentiate the accumulated short-rate integral only when required.
__device__ __forceinline__ float discount_factor(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float time
) {
    return expf(log_discount(parameters, state, time));
}

// Price one zero-coupon from the conditional Gaussian rate integral.
__device__ __forceinline__ float zero_coupon_bond(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float maturity
) {
    const float delta = maturity - valuation_time;
    const float loading = integral_factor_loading(
        parameters.dynamics.mean_reversion, delta
    );
    return expf(
        -loading * state.factor
        + 0.5f * integral_variance(parameters.dynamics, delta)
    );
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
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
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
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
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
) {
    const float start_discount = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    const float end_discount = zero_coupon_bond(
        parameters, state, valuation_time, end_time
    );
    return (start_discount / end_discount - 1.0f) / accrual_period;
}

// Divide the conditional floating-leg value by the fixed-leg annuity.
__device__ __forceinline__ float swap_rate(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
) {
    float annuity = 0.0f;
    for (std::size_t payment = 0U;
         payment < payment_count;
         ++payment) {
        annuity = fmaf(
            accrual_periods[payment],
            zero_coupon_bond(
                parameters,
                state,
                valuation_time,
                payment_times[payment]
            ),
            annuity
        );
    }
    const float start_discount = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    const float end_discount = zero_coupon_bond(
        parameters,
        state,
        valuation_time,
        payment_times[payment_count - 1U]
    );
    return (start_discount - end_discount) / annuity;
}

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
