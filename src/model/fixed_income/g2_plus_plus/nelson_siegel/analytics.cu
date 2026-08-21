// G2++ analytics composed from G2 and Nelson-Siegel formulas.
#pragma once

#include "model/fixed_income/g2_plus_plus/nelson_siegel/analytics.cuh"

// Include implementations so NVCC can inline process and curve formulas.
#include "curve/nelson_siegel/term_structure.cu"
#include "model/fixed_income/g2/dynamics.cu"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel {

// ======================= Model-specific analytics =========================

// Compose the curve-independent process with the fitted initial curve.
__device__ __forceinline__ G2PlusPlusFittedParameters compose_model(
    const ModelParameters& parameters,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve
) {
    return {parameters.process, initial_curve};
}

// Evaluate the deterministic shift fitting the initial forward curve.
__device__ __forceinline__ float short_rate_shift(
    const G2PlusPlusFittedParameters& parameters,
    float time
) {
    const float a = parameters.process.mean_reversion_x;
    const float b = parameters.process.mean_reversion_y;
    const float one_minus_decay_x = -expm1f(-a * time);
    const float one_minus_decay_y = -expm1f(-b * time);
    const float correction_x =
        parameters.process.volatility_x * parameters.process.volatility_x
        * one_minus_decay_x * one_minus_decay_x / (2.0f * a * a);
    const float correction_y =
        parameters.process.volatility_y * parameters.process.volatility_y
        * one_minus_decay_y * one_minus_decay_y / (2.0f * b * b);
    const float cross_correction = parameters.process.correlation
        * parameters.process.volatility_x * parameters.process.volatility_y
        * one_minus_decay_x * one_minus_decay_y / (a * b);
    return curve::nelson_siegel::instantaneous_forward(
        parameters.initial_curve, time
    ) + correction_x + correction_y + cross_correction;
}

// Add both Gaussian states to the deterministic curve shift.
__device__ __forceinline__ float short_rate(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
    float time
) {
    return state.state_x + state.state_y
        + short_rate_shift(parameters, time);
}

namespace {

constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

// Evaluate the standard normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    return 0.5f * erfcf(-value * kInverseSqrtTwo);
}

// Integrate the curve and convexity shift over [start,end].
__device__ __forceinline__ float shift_integral(
    const G2PlusPlusFittedParameters& parameters,
    float start,
    float end
) {
    const float forward_integral =
        curve::nelson_siegel::log_discount_factor(
            parameters.initial_curve, start
        )
        - curve::nelson_siegel::log_discount_factor(
            parameters.initial_curve, end
        );
    const float start_variance =
        model::g2::integral_moments(parameters.process, start).variance;
    const float end_variance =
        model::g2::integral_moments(parameters.process, end).variance;
    return forward_integral + 0.5f * (end_variance - start_variance);
}

struct AffineBondCoefficients {
    float log_A;
    model::g2::G2BondLoadings B;
};

// Compute fitted log(A), B_x, and B_y from one integral-moment evaluation.
__device__ __forceinline__ AffineBondCoefficients affine_bond_coefficients(
    const G2PlusPlusFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    const model::g2::IntegralMoments moments = model::g2::integral_moments(
        parameters.process, maturity - valuation_time
    );
    if (valuation_time == 0.0f) {
        return {
            curve::nelson_siegel::log_discount_factor(
                parameters.initial_curve, maturity
            ),
            {moments.state_x_loading, moments.state_y_loading},
        };
    }
    return {
        -shift_integral(parameters, valuation_time, maturity)
            + 0.5f * moments.variance,
        {moments.state_x_loading, moments.state_y_loading},
    };
}

// Return the two-factor log-forward volatility of one bond option.
__device__ __forceinline__ float bond_option_total_volatility(
    const model::g2::ProcessParameters& parameters,
    float time_to_expiry,
    float bond_tenor
) {
    const float a = parameters.mean_reversion_x;
    const float b = parameters.mean_reversion_y;
    const float variance_x = parameters.volatility_x * parameters.volatility_x
        * (-expm1f(-2.0f * a * time_to_expiry)) / (2.0f * a);
    const float variance_y = parameters.volatility_y * parameters.volatility_y
        * (-expm1f(-2.0f * b * time_to_expiry)) / (2.0f * b);
    const float covariance_xy = parameters.correlation
        * parameters.volatility_x * parameters.volatility_y
        * (-expm1f(-(a + b) * time_to_expiry)) / (a + b);
    const float loading_x =
        model::mean_reverting_gaussian::integral_state_loading(
            a, bond_tenor
        );
    const float loading_y =
        model::mean_reverting_gaussian::integral_state_loading(
            b, bond_tenor
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
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
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

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the fitted affine prefactor.
__device__ __forceinline__ float log_A(
    const G2PlusPlusFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters, valuation_time, maturity
    ).log_A;
}

// Exponentiate the affine prefactor only for callers requesting A itself.
__device__ __forceinline__ float A(
    const G2PlusPlusFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    return expf(log_A(parameters, valuation_time, maturity));
}

// Return both Gaussian-factor loadings in one value.
__device__ __forceinline__ model::g2::G2BondLoadings B(
    const G2PlusPlusFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    const float delta = maturity - valuation_time;
    return {
        model::mean_reverting_gaussian::integral_state_loading(
            parameters.process.mean_reversion_x, delta
        ),
        model::mean_reverting_gaussian::integral_state_loading(
            parameters.process.mean_reversion_y, delta
        ),
    };
}

// Evaluate log(A)-B_x*x-B_y*y from one grouped coefficient calculation.
__device__ __forceinline__ float log_zero_coupon_bond(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
    float valuation_time,
    float maturity
) {
    const AffineBondCoefficients coefficients = affine_bond_coefficients(
        parameters, valuation_time, maturity
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

// Combine the stochastic integral with the analytical curve shift.
__device__ __forceinline__ float log_discount_factor(
    const G2PlusPlusFittedParameters& parameters,
    float state_integral,
    float time
) {
    return -state_integral
        - shift_integral(parameters, 0.0f, time);
}

// Exponentiate the exact path log-discount only when needed.
__device__ __forceinline__ float discount_factor(
    const G2PlusPlusFittedParameters& parameters,
    float state_integral,
    float time
) {
    return expf(log_discount_factor(parameters, state_integral, time));
}

// Price one zero-coupon from both conditional Gaussian states.
__device__ __forceinline__ float zero_coupon_bond(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
    float valuation_time,
    float maturity
) {
    return expf(log_zero_coupon_bond(
        parameters, state, valuation_time, maturity
    ));
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
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
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
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
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
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
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::State& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::uint32_t payment_count
) {
    float annuity = 0.0f;
    float end_bond = 0.0f;
    for (std::uint32_t payment = 0U; payment < payment_count; ++payment) {
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

}  // namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel
