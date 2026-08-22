// Closed-form fixed-income analytics for the standalone G2 model.
#pragma once

#include "model/fixed_income/g2/analytics.cuh"

#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/gaussian_bond_option.cuh"

// Include exact G2 moments used by the analytical formulas below.
#include "model/fixed_income/g2/dynamics.cu"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::g2 {

// ======================= Model-specific analytics =========================

// Add both Gaussian factor states to reconstruct the short rate.
__device__ __forceinline__ float short_rate(const State& state) {
    return state.state_x + state.state_y;
}

// ==================== Model-specific implementation =======================

namespace {

struct AffineBondCoefficients {
    float log_A;
    G2BondLoadings B;
};

// Compute log(A), B_x, and B_y from one shared integral-moment evaluation.
__device__ __forceinline__ AffineBondCoefficients affine_bond_coefficients(
    const ProcessParameters& parameters,
    float delta
) {
    const IntegralMoments moments = integral_moments(
        parameters, delta
    );
    return {
        0.5f * moments.variance,
        {moments.state_x_loading, moments.state_y_loading},
    };
}

// Return the conditional covariance matrix of both future factor states.
__device__ __forceinline__ void state_covariances(
    const ProcessParameters& parameters,
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
    const ProcessParameters& parameters,
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
        mean_reverting_gaussian::integral_state_loading(
            parameters.mean_reversion_x, bond_tenor
        );
    const float loading_y =
        mean_reverting_gaussian::integral_state_loading(
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
    const ModelParameters& parameters,
    const State& state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    const float expiry_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time, option_expiry
    );
    const float underlying_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time, bond_maturity
    );
    const float expiry_bond = expf(expiry_log_bond);
    const float total_volatility = bond_option_total_volatility(
        parameters.process,
        option_expiry - valuation_time,
        bond_maturity - option_expiry
    );
    return fixed_income::discounted_lognormal_bond_option_price(
        {expiry_log_bond, expiry_bond},
        underlying_log_bond,
        total_volatility,
        strike,
        option_sign
    );
}


// Bind the two-factor bond function to common cashflow formulas.
struct AnalyticsProvider {
    __device__ __forceinline__ float zero_coupon_bond(
        const ModelParameters& parameters,
        const State& state,
        float valuation_time,
        float maturity
    ) const {
        return g2::zero_coupon_bond(
            parameters, state, valuation_time, maturity
        );
    }
};

}  // namespace

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the multiplicative affine prefactor.
__device__ __forceinline__ float log_A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    ).log_A;
}

// Exponentiate the affine prefactor only for callers requesting A itself.
__device__ __forceinline__ float A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return expf(log_A(parameters, valuation_time, maturity));
}

// Return both factor loadings in one value.
__device__ __forceinline__ G2BondLoadings B(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    const float delta = maturity - valuation_time;
    return {
        mean_reverting_gaussian::integral_state_loading(
            parameters.process.mean_reversion_x, delta
        ),
        mean_reverting_gaussian::integral_state_loading(
            parameters.process.mean_reversion_y, delta
        ),
    };
}

// Evaluate log(A)-B_x*x-B_y*y from one grouped coefficient calculation.
__device__ __forceinline__ float log_zero_coupon_bond(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float maturity
) {
    const AffineBondCoefficients coefficients = affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    );
    return fmaf(
        -coefficients.B.state_x,
        state.state_x,
        fmaf(
            -coefficients.B.state_y,
            state.state_y,
            coefficients.log_A
        )
    );
}

// The joint integral is the accumulated standalone G2 short rate.
__device__ __forceinline__ float log_discount_factor(
    float state_integral
) {
    return -state_integral;
}

// Exponentiate the accumulated rate integral only when required.
__device__ __forceinline__ float discount_factor(
    float state_integral
) {
    return expf(log_discount_factor(state_integral));
}

// Price one zero-coupon from the conditional Gaussian rate integral.
__device__ __forceinline__ float zero_coupon_bond(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float maturity
) {
    return expf(log_zero_coupon_bond(
        parameters, state, valuation_time, maturity
    ));
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const ModelParameters& parameters,
    const State& state,
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
    const ModelParameters& parameters,
    const State& state,
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
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
) {
    return fixed_income::forward_rate(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time,
        start_time,
        end_time,
        accrual_period
    );
}

template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    const ScheduleView& schedule
) {
    return fixed_income::swap_rate(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time,
        start_time,
        schedule
    );
}

}  // namespace ai_factory::workbench::model::g2
