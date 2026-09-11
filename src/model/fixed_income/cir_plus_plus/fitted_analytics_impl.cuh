// Included device analytics definitions for CIR++ shifts, bonds and bond options.
// Jamshidian cashflows and noncentral-chi-square tails remain shared primitives.
#pragma once

#include "model/fixed_income/cir_plus_plus/fitted_analytics.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "common/fixed_income/jamshidian.cuh"

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::fitted {

template<typename CurveProvider>
__device__ __forceinline__ FittedParameters<CurveProvider> compose_fitted_model(
    const ModelParameters& factor, const typename CurveProvider::Parameters& curve
) {
    return {factor, curve};
}

// phi(t) = f_market(0,t) - f_CIR(0,t). The derivative of B is evaluated
// without subtracting Riccati terms that cancel at long maturities.
template<typename CurveProvider>
__device__ __forceinline__ float short_rate_shift(
    const FittedParameters<CurveProvider>& parameters, float time_years
) {
    const auto& factor = parameters.factor;
    const float kappa = factor.process.mean_reversion;
    const float sigma = factor.process.volatility;
    const float gamma = sqrtf(kappa * kappa + 2.0f * sigma * sigma);
    const float gamma_minus_kappa = 2.0f * sigma * sigma / (gamma + kappa);
    const float decay = expf(-gamma * time_years);
    const float denominator = fmaf(-gamma_minus_kappa, -expm1f(-gamma * time_years),
        2.0f * gamma);
    const float ratio = 2.0f * gamma / denominator;
    const float loading_derivative = decay * ratio * ratio;
    const float initial_forward = fmaf(factor.initial_state, loading_derivative,
        kappa * factor.process.long_term_mean * cir::B(factor, 0.0f, time_years));
    return CurveProvider::instantaneous_forward(parameters.initial_curve, time_years)
        - initial_forward;
}

template<typename CurveProvider>
__device__ __forceinline__ float short_rate(
    const FittedParameters<CurveProvider>& parameters, float state, float time_years
) {
    return state + short_rate_shift(parameters, time_years);
}

// Integral of the deterministic shift, expressed entirely through log bonds.
template<typename CurveProvider>
__device__ __forceinline__ float shift_integral(
    const FittedParameters<CurveProvider>& parameters, float start_years, float end_years
) {
    const auto& factor = parameters.factor;
    return CurveProvider::log_discount_factor(parameters.initial_curve, start_years)
        - CurveProvider::log_discount_factor(parameters.initial_curve, end_years)
        - cir::log_zero_coupon_bond(factor, factor.initial_state, 0.0f, start_years)
        + cir::log_zero_coupon_bond(factor, factor.initial_state, 0.0f, end_years);
}

template<typename CurveProvider>
__device__ __forceinline__ ::ai_factory::workbench::fixed_income::OneFactorAffineBondCoefficients
affine_bond_coefficients(
    const FittedParameters<CurveProvider>& parameters, float time_years, float maturity_years
) {
    auto coefficients = cir::AnalyticsProvider{}.affine_bond_coefficients(
        parameters.factor, time_years, maturity_years
    );
    coefficients.log_A -= shift_integral(parameters, time_years, maturity_years);
    return coefficients;
}

template<typename CurveProvider>
__device__ __forceinline__ ::ai_factory::workbench::fixed_income::OneFactorAffineBondCoefficients
AnalyticsProvider<CurveProvider>::affine_bond_coefficients(
    const Parameters& parameters, float time_years, float maturity_years
) const {
    return fitted::affine_bond_coefficients(parameters, time_years, maturity_years);
}

template<typename CurveProvider>
__device__ __forceinline__ float AnalyticsProvider<CurveProvider>::zero_coupon_bond(
    const Parameters& parameters, float state, float time_years, float maturity_years
) const {
    return ::ai_factory::workbench::fixed_income::zero_coupon_bond(
        *this, parameters, state, time_years, maturity_years
    );
}

template<typename CurveProvider>
__device__ __forceinline__ typename AnalyticsProvider<CurveProvider>::BondOptionContext
AnalyticsProvider<CurveProvider>::prepare_bond_option_context(
    const Parameters& parameters, float state, float time_years, float expiry_years
) const {
    return cir::AnalyticsProvider{}.prepare_bond_option_context(
        parameters.factor, state, time_years, expiry_years
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float AnalyticsProvider<CurveProvider>::bond_option_price(
    const BondOptionContext& context, const Parameters& parameters, float state,
    float sign, float time_years, float expiry_years, float maturity_years, float strike
) const {
    // D(t,T) * CIR_option(K / D(S,T)); prepare the common CIR CDF context once
    // per expiry_years, including when Jamshidian evaluates a whole cashflow strip.
    const float adjusted_strike = strike * expf(shift_integral(parameters, expiry_years, maturity_years));
    return expf(-shift_integral(parameters, time_years, maturity_years))
        * cir::AnalyticsProvider{}.bond_option_price(context, parameters.factor,
            state, sign, time_years, expiry_years, maturity_years, adjusted_strike);
}

template<typename CurveProvider>
__device__ __forceinline__ float log_A(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
) {
    return affine_bond_coefficients(
        parameters, valuation_time_years, maturity_years
    ).log_A;
}

template<typename CurveProvider>
__device__ __forceinline__ float A(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
) {
    return expf(log_A(parameters, valuation_time_years, maturity_years));
}

template<typename CurveProvider>
__device__ __forceinline__ float B(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
) {
    return affine_bond_coefficients(
        parameters, valuation_time_years, maturity_years
    ).B;
}

template<typename CurveProvider>
__device__ __forceinline__ float log_zero_coupon_bond(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float maturity_years
) {
    return ::ai_factory::workbench::fixed_income::log_zero_coupon_bond(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time_years,
        maturity_years
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float log_discount_factor(
    const FittedParameters<CurveProvider>& parameters,
    float state_integral,
    float time_years
) {
    return -state_integral - shift_integral(parameters, 0.0f, time_years);
}

template<typename CurveProvider>
__device__ __forceinline__ float discount_factor(
    const FittedParameters<CurveProvider>& parameters,
    float state_integral,
    float time_years
) {
    return expf(log_discount_factor(parameters, state_integral, time_years));
}

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float maturity_years
) {
    return ::ai_factory::workbench::fixed_income::zero_coupon_bond(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time_years,
        maturity_years
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float option_sign,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
) {
    const AnalyticsProvider<CurveProvider> provider{};
    return provider.bond_option_price(
        provider.prepare_bond_option_context(
            parameters, state, valuation_time_years, option_expiry_years
        ),
        parameters,
        state,
        option_sign,
        valuation_time_years,
        option_expiry_years,
        bond_maturity_years,
        strike
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
) {
    return zero_coupon_bond_option_price(
        parameters,
        state,
        1.0f,
        valuation_time_years,
        option_expiry_years,
        bond_maturity_years,
        strike
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
) {
    return zero_coupon_bond_option_price(
        parameters,
        state,
        -1.0f,
        valuation_time_years,
        option_expiry_years,
        bond_maturity_years,
        strike
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float forward_rate(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    float end_time_years,
    float accrual_fraction
) {
    return ::ai_factory::workbench::fixed_income::forward_rate(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time_years,
        start_time_years,
        end_time_years,
        accrual_fraction
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::swap_rate(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time_years,
        start_time_years,
        schedule
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::payer_swap_value(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time_years,
        start_time_years,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float jamshidian_state_boundary(
    const FittedParameters<CurveProvider>& parameters,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::jamshidian_state_boundary(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float jamshidian_state_boundary(
    const FittedParameters<CurveProvider>& parameters,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times_days,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
) {
    return jamshidian_state_boundary(
        parameters,
        exercise_time,
        fixed_rate,
        ::ai_factory::workbench::fixed_income::BusinessDayFixedLegScheduleView{
            payment_times_days,
            accrual_fractions,
            payment_count,
            time_day_fraction,
        }
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float jamshidian_bond_strike(
    const FittedParameters<CurveProvider>& parameters,
    float exercise_time,
    float payment_time,
    float state_boundary
) {
    return ::ai_factory::workbench::fixed_income::jamshidian_bond_strike(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        exercise_time,
        payment_time,
        state_boundary
    );
}

template<SwaptionSide Side, typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float european_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::european_swaption_price<Side>(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time_years,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float european_payer_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return european_swaption_price<SwaptionSide::payer>(
        parameters,
        state,
        valuation_time_years,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float european_payer_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times_days,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
) {
    return european_payer_swaption_price(
        parameters,
        state,
        valuation_time_years,
        exercise_time,
        fixed_rate,
        ::ai_factory::workbench::fixed_income::BusinessDayFixedLegScheduleView{
            payment_times_days,
            accrual_fractions,
            payment_count,
            time_day_fraction,
        }
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float european_receiver_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return european_swaption_price<SwaptionSide::receiver>(
        parameters,
        state,
        valuation_time_years,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float european_receiver_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times_days,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
) {
    return european_receiver_swaption_price(
        parameters,
        state,
        valuation_time_years,
        exercise_time,
        fixed_rate,
        ::ai_factory::workbench::fixed_income::BusinessDayFixedLegScheduleView{
            payment_times_days,
            accrual_fractions,
            payment_count,
            time_day_fraction,
        }
    );
}

}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::fitted
