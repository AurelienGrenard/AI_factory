// Closed-form fixed-income analytics for the Vasicek model.
#pragma once

#include "model/fixed_income/vasicek/analytics.cuh"

#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/gaussian_bond_option.cuh"
#include "common/fixed_income/jamshidian.cuh"
#include "common/fixed_income/one_factor_affine.cuh"

// Include exact Vasicek moments and transitions used by the analytics below.
#include "model/fixed_income/vasicek/dynamics.cu"

#include <cuda_runtime.h>

#include <cfloat>

namespace ai_factory::workbench::model::vasicek {

// ==================== Model-specific implementation =======================

namespace {

using AffineBondCoefficients =
    fixed_income::OneFactorAffineBondCoefficients;

struct BondOptionContext {
    fixed_income::GaussianBondOptionDiscountContext discount;
    float expiry_state_standard_deviation;
};

// Compute log(A) and B together from one shared set of integral moments.
__device__ __forceinline__ AffineBondCoefficients affine_bond_coefficients(
    const ProcessParameters& parameters,
    float delta
) {
    const IntegralMoments moments = integral_moments(
        parameters, delta
    );
    return {
        0.5f * moments.variance - moments.mean_increment,
        moments.state_loading,
    };
}

// Prepare all model/expiry invariants once for a strip of bond options.
__device__ __forceinline__ BondOptionContext prepare_bond_option_context(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry
) {
    const float time_to_expiry = option_expiry - valuation_time;
    const IntegralMoments expiry_moments = integral_moments(
        parameters.process, time_to_expiry
    );
    const float expiry_log_bond = fmaf(
        -expiry_moments.state_loading,
        state,
        0.5f * expiry_moments.variance - expiry_moments.mean_increment
    );
    const float expiry_bond = expf(expiry_log_bond);

    const float a = parameters.process.mean_reversion;
    const float one_minus_decay = a * expiry_moments.state_loading;
    const float expiry_state_variance =
        parameters.process.volatility * parameters.process.volatility
        * expiry_moments.state_loading * (1.0f - 0.5f * one_minus_decay);
    return {
        {expiry_log_bond, expiry_bond},
        sqrtf(fmaxf(expiry_state_variance, 0.0f)),
    };
}

// Price one bond option while reusing the prepared expiry context.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const BondOptionContext& context,
    const ModelParameters& parameters,
    float state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    const float underlying_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time, bond_maturity
    );
    const float bond_loading = integral_state_loading(
        parameters.process.mean_reversion,
        bond_maturity - option_expiry
    );
    return fixed_income::discounted_lognormal_bond_option_price(
        context.discount,
        underlying_log_bond,
        bond_loading * context.expiry_state_standard_deviation,
        strike,
        option_sign
    );
}

// Adapt a standalone bond option to the strip-oriented implementation.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const ModelParameters& parameters,
    float state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    return zero_coupon_bond_option_price(
        prepare_bond_option_context(
            parameters, state, valuation_time, option_expiry
        ),
        parameters,
        state,
        option_sign,
        valuation_time,
        option_expiry,
        bond_maturity,
        strike
    );
}

// Bind model-specific A/B and Gaussian option inputs to common formulas.
struct AnalyticsProvider {
    __device__ __forceinline__ AffineBondCoefficients
    affine_bond_coefficients(
        const ModelParameters& parameters,
        float valuation_time,
        float maturity
    ) const {
        return vasicek::affine_bond_coefficients(
            parameters.process, maturity - valuation_time
        );
    }

    __device__ __forceinline__ float zero_coupon_bond(
        const ModelParameters& parameters,
        float state,
        float valuation_time,
        float maturity
    ) const {
        return fixed_income::zero_coupon_bond(
            *this, parameters, state, valuation_time, maturity
        );
    }

    __device__ __forceinline__ BondOptionContext
    prepare_bond_option_context(
        const ModelParameters& parameters,
        float state,
        float valuation_time,
        float option_expiry
    ) const {
        return vasicek::prepare_bond_option_context(
            parameters, state, valuation_time, option_expiry
        );
    }

    __device__ __forceinline__ float bond_option_price(
        const BondOptionContext& context,
        const ModelParameters& parameters,
        float state,
        float option_sign,
        float valuation_time,
        float option_expiry,
        float bond_maturity,
        float strike
    ) const {
        return vasicek::zero_coupon_bond_option_price(
            context,
            parameters,
            state,
            option_sign,
            valuation_time,
            option_expiry,
            bond_maturity,
            strike
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

// Return the current-rate loading in the affine bond expression.
__device__ __forceinline__ float B(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return integral_state_loading(
        parameters.process.mean_reversion, maturity - valuation_time
    );
}

// Evaluate log(A)-B*r from one grouped coefficient calculation.
__device__ __forceinline__ float log_zero_coupon_bond(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return fixed_income::log_zero_coupon_bond(
        AnalyticsProvider{}, parameters, state, valuation_time, maturity
    );
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
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return fixed_income::zero_coupon_bond(
        AnalyticsProvider{}, parameters, state, valuation_time, maturity
    );
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const ModelParameters& parameters,
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
    const ModelParameters& parameters,
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
    const ModelParameters& parameters,
    float state,
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

// Divide the floating-leg value by the fixed-leg annuity.
template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const ModelParameters& parameters,
    float state,
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

// Value the floating leg minus the fixed leg at unit notional.
template<typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return fixed_income::payer_swap_value(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time,
        start_time,
        fixed_rate,
        schedule
    );
}

// Locate the monotone coupon-bond boundary for either schedule layout.
template<typename ScheduleView>
__device__ __forceinline__ float jamshidian_state_boundary(
    const ModelParameters& parameters,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return fixed_income::jamshidian_state_boundary(
        AnalyticsProvider{},
        parameters,
        exercise_time,
        fixed_rate,
        schedule
    );
}

__device__ __forceinline__ float jamshidian_state_boundary(
    const ModelParameters& parameters,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
) {
    return jamshidian_state_boundary(
        parameters,
        exercise_time,
        fixed_rate,
        product::ExplicitEuropeanSwaptionScheduleView{
            payment_times,
            accrual_fractions,
            payment_count,
            time_day_fraction,
        }
    );
}

__device__ __forceinline__ float jamshidian_bond_strike(
    const ModelParameters& parameters,
    float exercise_time,
    float payment_time,
    float state_boundary
) {
    return fixed_income::jamshidian_bond_strike(
        AnalyticsProvider{},
        parameters,
        exercise_time,
        payment_time,
        state_boundary
    );
}

template<SwaptionSide Side, typename ScheduleView>
__device__ __forceinline__ float european_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return fixed_income::european_swaption_price<Side>(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename ScheduleView>
__device__ __forceinline__ float european_payer_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return european_swaption_price<SwaptionSide::payer>(
        parameters, state, valuation_time, exercise_time, fixed_rate, schedule
    );
}

__device__ __forceinline__ float european_payer_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
) {
    return european_payer_swaption_price(
        parameters, state, valuation_time, exercise_time, fixed_rate,
        product::ExplicitEuropeanSwaptionScheduleView{
            payment_times, accrual_fractions, payment_count, time_day_fraction,
        }
    );
}

template<typename ScheduleView>
__device__ __forceinline__ float european_receiver_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return european_swaption_price<SwaptionSide::receiver>(
        parameters, state, valuation_time, exercise_time, fixed_rate, schedule
    );
}

__device__ __forceinline__ float european_receiver_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
) {
    return european_receiver_swaption_price(
        parameters, state, valuation_time, exercise_time, fixed_rate,
        product::ExplicitEuropeanSwaptionScheduleView{
            payment_times, accrual_fractions, payment_count, time_day_fraction,
        }
    );
}

}  // namespace ai_factory::workbench::model::vasicek
