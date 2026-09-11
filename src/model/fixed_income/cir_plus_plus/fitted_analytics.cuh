// Curve-independent CIR++ composition and public affine analytics contracts.
#pragma once

#include "common/fixed_income/analytics_concepts.cuh"
#include "model/fixed_income/cir/analytics.cuh"
#include "model/fixed_income/cir_plus_plus/parameters.hpp"

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::fitted {

template<typename CurveProvider>
struct FittedParameters {
    ModelParameters factor;
    typename CurveProvider::Parameters initial_curve;
};

template<typename CurveProvider>
struct AnalyticsProvider {
    using Parameters = FittedParameters<CurveProvider>;
    using BondOptionContext = cir::BondOptionContext;

    __device__ __forceinline__ ::ai_factory::workbench::fixed_income::OneFactorAffineBondCoefficients
    affine_bond_coefficients(const Parameters&, float time_years, float maturity_years) const;
    __device__ __forceinline__ float zero_coupon_bond(
        const Parameters&, float state, float time_years, float maturity_years
    ) const;
    __device__ __forceinline__ BondOptionContext prepare_bond_option_context(
        const Parameters&, float state, float time_years, float expiry_years
    ) const;
    __device__ __forceinline__ float bond_option_price(
        const BondOptionContext&, const Parameters&, float state, float sign,
        float time_years, float expiry_years, float maturity_years, float strike
    ) const;
};

template<typename CurveProvider>
__device__ __forceinline__ FittedParameters<CurveProvider> compose_fitted_model(
    const ModelParameters& factor, const typename CurveProvider::Parameters& curve
);

template<typename CurveProvider>
__device__ __forceinline__ float short_rate_shift(
    const FittedParameters<CurveProvider>& parameters, float time_years
);

template<typename CurveProvider>
__device__ __forceinline__ float short_rate(
    const FittedParameters<CurveProvider>& parameters, float state, float time_years
);

template<typename CurveProvider>
__device__ __forceinline__ float shift_integral(
    const FittedParameters<CurveProvider>& parameters, float start_years, float end_years
);

template<typename CurveProvider>
__device__ __forceinline__ ::ai_factory::workbench::fixed_income::OneFactorAffineBondCoefficients
affine_bond_coefficients(
    const FittedParameters<CurveProvider>& parameters, float time_years, float maturity_years
);

template<typename CurveProvider>
__device__ __forceinline__ float log_A(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
);

template<typename CurveProvider>
__device__ __forceinline__ float A(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
);

template<typename CurveProvider>
__device__ __forceinline__ float B(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
);

template<typename CurveProvider>
__device__ __forceinline__ float log_zero_coupon_bond(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float maturity_years
);

template<typename CurveProvider>
__device__ __forceinline__ float log_discount_factor(
    const FittedParameters<CurveProvider>& parameters,
    float state_integral,
    float time_years
);

template<typename CurveProvider>
__device__ __forceinline__ float discount_factor(
    const FittedParameters<CurveProvider>& parameters,
    float state_integral,
    float time_years
);

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float maturity_years
);

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float option_sign,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
);

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
);

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
);

template<typename CurveProvider>
__device__ __forceinline__ float forward_rate(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    float end_time_years,
    float accrual_fraction
);

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    const ScheduleView& schedule
);

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    float fixed_rate,
    const ScheduleView& schedule
);

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float jamshidian_state_boundary(
    const FittedParameters<CurveProvider>& parameters,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

template<typename CurveProvider>
__device__ __forceinline__ float jamshidian_state_boundary(
    const FittedParameters<CurveProvider>& parameters,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times_days,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
);

template<typename CurveProvider>
__device__ __forceinline__ float jamshidian_bond_strike(
    const FittedParameters<CurveProvider>& parameters,
    float exercise_time,
    float payment_time,
    float state_boundary
);

template<SwaptionSide Side, typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float european_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float european_payer_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

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
);

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float european_receiver_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

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
);

// The normalized regression factor is exactly the standalone CIR factor.
template<typename Composition>
struct BermudanSwaptionAnalyticsPolicy : cir::BermudanSwaptionAnalyticsPolicy {
    using PreparedModel = typename Composition::FittedModel;
    using CurveParameters = typename Composition::CurveParameters;
    __device__ __forceinline__ static PreparedModel prepare_model(
        const ModelParameters& factor, const CurveParameters& curve
    ) { return Composition::compose(factor, curve); }

    __device__ __forceinline__ static float log_discount_factor(
        const PreparedModel& model, float state_integral, float time_years
    ) { return fitted::log_discount_factor(model, state_integral, time_years); }

    template<typename ScheduleView>
    __device__ __forceinline__ static float payer_swap_value(
        const PreparedModel& model, float state, float time_years, float start_years,
        float fixed_rate, const ScheduleView& schedule
    ) {
        return fitted::payer_swap_value(model, state, time_years, start_years, fixed_rate, schedule);
    }
};

// A deterministic shift cancels in the normalized terminal-bond density:
// CIR++ uses the CIR terminal-forward law but its own fitted numeraire value.
template<typename Composition>
struct TerminalForwardBondAnalyticsPolicy {
    using Parameters = typename Composition::FittedModel;
    __device__ __forceinline__ static float log_zero_coupon_bond(
        const Parameters& model, float state, float time_years, float maturity_years
    ) { return fitted::log_zero_coupon_bond(model, state, time_years, maturity_years); }
    __device__ __forceinline__ static float log_A(
        const Parameters& model, float time_years, float maturity_years
    ) { return fitted::log_A(model, time_years, maturity_years); }
    __device__ __forceinline__ static float B(
        const Parameters& model, float time_years, float maturity_years
    ) { return fitted::B(model, time_years, maturity_years); }
};

}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::fitted
