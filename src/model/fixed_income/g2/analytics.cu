// Closed-form fixed-income analytics for the standalone G2 model.
#pragma once

#include "model/fixed_income/g2/analytics.cuh"

// Include exact G2 moments used by the analytical formulas below.
#include "model/fixed_income/g2/dynamics.cu"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::g2 {
namespace {

constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

// Evaluate the standard normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    return 0.5f * erfcf(-value * kInverseSqrtTwo);
}

// Return the conditional log price of one zero-coupon bond.
__device__ __forceinline__ float log_zero_coupon_bond(
    const G2ModelParameters& parameters,
    const G2State& state,
    float delta
) {
    const G2IntegralMoments moments = integral_moments(
        parameters.process, delta
    );
    return -moments.state_x_loading * state.state_x
        - moments.state_y_loading * state.state_y
        + 0.5f * moments.variance;
}

// Return the conditional covariance matrix of both future factor states.
__device__ __forceinline__ void state_covariances(
    const G2ProcessParameters& parameters,
    float delta,
    float& variance_x,
    float& variance_y,
    float& covariance_xy
) {
    const float a = parameters.mean_reversion_x;
    const float b = parameters.mean_reversion_y;
    variance_x = parameters.volatility_x * parameters.volatility_x
        * (-expm1f(-2.0f * a * delta)) / (2.0f * a);
    variance_y = parameters.volatility_y * parameters.volatility_y
        * (-expm1f(-2.0f * b * delta)) / (2.0f * b);
    covariance_xy = parameters.correlation
        * parameters.volatility_x * parameters.volatility_y
        * (-expm1f(-(a + b) * delta)) / (a + b);
}

// Return the total log-forward volatility of one bond option.
__device__ __forceinline__ float bond_option_total_volatility(
    const G2ProcessParameters& parameters,
    float time_to_expiry,
    float bond_tenor
) {
    float variance_x = 0.0f;
    float variance_y = 0.0f;
    float covariance_xy = 0.0f;
    state_covariances(
        parameters,
        time_to_expiry,
        variance_x,
        variance_y,
        covariance_xy
    );
    const float loading_x =
        model::ornstein_uhlenbeck::integral_state_loading(
            parameters.mean_reversion_x, bond_tenor
        );
    const float loading_y =
        model::ornstein_uhlenbeck::integral_state_loading(
            parameters.mean_reversion_y, bond_tenor
        );
    const float variance = fmaf(
        loading_x * loading_x,
        variance_x,
        fmaf(
            loading_y * loading_y,
            variance_y,
            2.0f * loading_x * loading_y * covariance_xy
        )
    );
    return sqrtf(fmaxf(variance, 0.0f));
}

// Price a call (+1) or put (-1) with one shared Black-style expression.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const G2ModelParameters& parameters,
    const G2State& state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    const float expiry_log_bond = log_zero_coupon_bond(
        parameters, state, option_expiry - valuation_time
    );
    const float underlying_log_bond = log_zero_coupon_bond(
        parameters, state, bond_maturity - valuation_time
    );
    const float expiry_bond = expf(expiry_log_bond);
    const float underlying_bond = expf(underlying_log_bond);
    const float total_volatility = bond_option_total_volatility(
        parameters.process,
        option_expiry - valuation_time,
        bond_maturity - option_expiry
    );
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
                - strike * expiry_bond * normal_cdf(option_sign * d2)
            ),
        0.0f
    );
}

}  // namespace

// Add both Gaussian factor states to reconstruct the short rate.
__device__ __forceinline__ float short_rate(const G2State& state) {
    return state.state_x + state.state_y;
}

// The joint integral is the accumulated standalone G2 short rate.
__device__ __forceinline__ float log_discount_factor(
    const joint::G2JointState& joint_state
) {
    return -joint_state.state_integral;
}

// Exponentiate the accumulated rate integral only when required.
__device__ __forceinline__ float discount_factor(
    const joint::G2JointState& joint_state
) {
    return expf(log_discount_factor(joint_state));
}

// Price one zero-coupon from the conditional Gaussian rate integral.
__device__ __forceinline__ float zero_coupon_bond(
    const G2ModelParameters& parameters,
    const G2State& state,
    float valuation_time,
    float maturity
) {
    return expf(log_zero_coupon_bond(
        parameters, state, maturity - valuation_time
    ));
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const G2ModelParameters& parameters,
    const G2State& state,
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
    const G2ModelParameters& parameters,
    const G2State& state,
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
    const G2ModelParameters& parameters,
    const G2State& state,
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
    const G2ModelParameters& parameters,
    const G2State& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
) {
    float annuity = 0.0f;
    float end_bond = 0.0f;
    for (std::size_t payment = 0U; payment < payment_count; ++payment) {
        const float current_bond = zero_coupon_bond(
            parameters, state, valuation_time, payment_times[payment]
        );
        annuity = fmaf(accrual_periods[payment], current_bond, annuity);
        end_bond = current_bond;
    }
    const float start_bond = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    return (start_bond - end_bond) / annuity;
}

}  // namespace ai_factory::workbench::model::g2
