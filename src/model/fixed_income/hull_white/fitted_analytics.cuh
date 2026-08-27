// Curve-agnostic Hull-White analytics built from the standalone OU process.
#pragma once

#include "common/fixed_income/analytics_concepts.cuh"
#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/gaussian_bond_option.cuh"
#include "common/fixed_income/jamshidian.cuh"
#include "common/fixed_income/mean_reverting_gaussian.cuh"
#include "model/fixed_income/hull_white/parameters.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/analytics.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::hull_white::fitted {

template<typename CurveProvider>
struct FittedParameters {
    model::fixed_income::ornstein_uhlenbeck::ProcessParameters process;
    typename CurveProvider::Parameters initial_curve;
};

template<typename CurveProvider>
struct BondOptionContext {
    ::ai_factory::workbench::fixed_income::GaussianBondOptionDiscountContext
        discount;
    float expiry_state_standard_deviation;
};

template<typename CurveProvider>
__device__ __forceinline__ FittedParameters<CurveProvider>
compose_fitted_model(
    const ModelParameters& parameters,
    const typename CurveProvider::Parameters& initial_curve
) {
    return {
        {parameters.mean_reversion, parameters.volatility},
        initial_curve,
    };
}

template<typename CurveProvider>
__device__ __forceinline__ float short_rate_shift(
    const FittedParameters<CurveProvider>& parameters,
    float time
) {
    const float mean_reversion = parameters.process.mean_reversion;
    const float one_minus_decay = -expm1f(-mean_reversion * time);
    const float correction =
        parameters.process.volatility * parameters.process.volatility
        * one_minus_decay * one_minus_decay
        / (2.0f * mean_reversion * mean_reversion);
    return CurveProvider::instantaneous_forward(
        parameters.initial_curve, time
    ) + correction;
}

template<typename CurveProvider>
__device__ __forceinline__ float short_rate(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float time
) {
    return state + short_rate_shift(parameters, time);
}

template<typename CurveProvider>
__device__ __forceinline__ float shift_integral(
    const FittedParameters<CurveProvider>& parameters,
    float start_time,
    float end_time
) {
    const float mean_reversion = parameters.process.mean_reversion;
    const float time_interval = end_time - start_time;
    const float one_minus_decay = -expm1f(
        -mean_reversion * time_interval
    );
    const float one_minus_decay_squared = -expm1f(
        -2.0f * mean_reversion * time_interval
    );
    const float forward_integral =
        CurveProvider::log_discount_factor(
            parameters.initial_curve, start_time
        )
        - CurveProvider::log_discount_factor(
            parameters.initial_curve, end_time
        );
    if (start_time == 0.0f) {
        return forward_integral
            + 0.5f
                * model::fixed_income::ornstein_uhlenbeck::integral_variance(
                    parameters.process, end_time
                );
    }
    const float convexity_integral =
        parameters.process.volatility * parameters.process.volatility
        / (2.0f * mean_reversion * mean_reversion)
        * (
            time_interval
            - 2.0f * expf(-mean_reversion * start_time)
                * one_minus_decay / mean_reversion
            + expf(-2.0f * mean_reversion * start_time)
                * one_minus_decay_squared / (2.0f * mean_reversion)
        );
    return forward_integral + convexity_integral;
}

template<typename CurveProvider>
__device__ __forceinline__
::ai_factory::workbench::fixed_income::OneFactorAffineBondCoefficients
affine_bond_coefficients(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time,
    float maturity
) {
    const auto moments =
        model::fixed_income::ornstein_uhlenbeck::integral_moments(
            parameters.process, maturity - valuation_time
        );
    if (valuation_time == 0.0f) {
        return {
            CurveProvider::log_discount_factor(
                parameters.initial_curve, maturity
            ),
            moments.state_loading,
        };
    }
    return {
        -shift_integral(parameters, valuation_time, maturity)
            + 0.5f * moments.variance,
        moments.state_loading,
    };
}

template<typename CurveProvider>
__device__ __forceinline__ BondOptionContext<CurveProvider>
prepare_bond_option_context(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
    float option_expiry
) {
    const auto coefficients = affine_bond_coefficients(
        parameters, valuation_time, option_expiry
    );
    const float expiry_log_bond = fmaf(
        -coefficients.B, state, coefficients.log_A
    );
    const float mean_reversion = parameters.process.mean_reversion;
    const float time_to_expiry = option_expiry - valuation_time;
    const float state_variance =
        parameters.process.volatility * parameters.process.volatility
        * (-expm1f(-2.0f * mean_reversion * time_to_expiry))
        / (2.0f * mean_reversion);
    return {
        {expiry_log_bond, expf(expiry_log_bond)},
        sqrtf(fmaxf(state_variance, 0.0f)),
    };
}

template<typename CurveProvider>
struct AnalyticsProvider {
    using Parameters = FittedParameters<CurveProvider>;

    __device__ __forceinline__
    ::ai_factory::workbench::fixed_income::OneFactorAffineBondCoefficients
    affine_bond_coefficients(
        const Parameters& parameters,
        float valuation_time,
        float maturity
    ) const {
        return fitted::affine_bond_coefficients(
            parameters, valuation_time, maturity
        );
    }

    __device__ __forceinline__ float zero_coupon_bond(
        const Parameters& parameters,
        float state,
        float valuation_time,
        float maturity
    ) const {
        return ::ai_factory::workbench::fixed_income::zero_coupon_bond(
            *this, parameters, state, valuation_time, maturity
        );
    }

    __device__ __forceinline__ BondOptionContext<CurveProvider>
    prepare_bond_option_context(
        const Parameters& parameters,
        float state,
        float valuation_time,
        float option_expiry
    ) const {
        return fitted::prepare_bond_option_context(
            parameters, state, valuation_time, option_expiry
        );
    }

    __device__ __forceinline__ float bond_option_price(
        const BondOptionContext<CurveProvider>& context,
        const Parameters& parameters,
        float state,
        float option_sign,
        float valuation_time,
        float option_expiry,
        float bond_maturity,
        float strike
    ) const {
        const auto coefficients = affine_bond_coefficients(
            parameters, valuation_time, bond_maturity
        );
        const float underlying_log_bond = fmaf(
            -coefficients.B, state, coefficients.log_A
        );
        const float bond_loading =
            model::fixed_income::ornstein_uhlenbeck::integral_state_loading(
                parameters.process.mean_reversion,
                bond_maturity - option_expiry
            );
        return ::ai_factory::workbench::fixed_income::
            discounted_lognormal_bond_option_price(
                context.discount,
                underlying_log_bond,
                bond_loading * context.expiry_state_standard_deviation,
                strike,
                option_sign
            );
    }
};

template<typename CurveProvider>
__device__ __forceinline__ float log_A(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters, valuation_time, maturity
    ).log_A;
}

template<typename CurveProvider>
__device__ __forceinline__ float A(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time,
    float maturity
) {
    return expf(log_A(parameters, valuation_time, maturity));
}

template<typename CurveProvider>
__device__ __forceinline__ float B(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters, valuation_time, maturity
    ).B;
}

template<typename CurveProvider>
__device__ __forceinline__ float log_zero_coupon_bond(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return ::ai_factory::workbench::fixed_income::log_zero_coupon_bond(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time,
        maturity
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float log_discount_factor(
    const FittedParameters<CurveProvider>& parameters,
    float state_integral,
    float time
) {
    return -state_integral - shift_integral(parameters, 0.0f, time);
}

template<typename CurveProvider>
__device__ __forceinline__ float discount_factor(
    const FittedParameters<CurveProvider>& parameters,
    float state_integral,
    float time
) {
    return expf(log_discount_factor(parameters, state_integral, time));
}

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return ::ai_factory::workbench::fixed_income::zero_coupon_bond(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time,
        maturity
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    const AnalyticsProvider<CurveProvider> provider{};
    return provider.bond_option_price(
        provider.prepare_bond_option_context(
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

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const FittedParameters<CurveProvider>& parameters,
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

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const FittedParameters<CurveProvider>& parameters,
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

template<typename CurveProvider>
__device__ __forceinline__ float forward_rate(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_fraction
) {
    return ::ai_factory::workbench::fixed_income::forward_rate(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time,
        start_time,
        end_time,
        accrual_fraction
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
    float start_time,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::swap_rate(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time,
        start_time,
        schedule
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
    float start_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::payer_swap_value(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time,
        start_time,
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
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::european_swaption_price<Side>(
        AnalyticsProvider<CurveProvider>{},
        parameters,
        state,
        valuation_time,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider, typename ScheduleView>
__device__ __forceinline__ float european_payer_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return european_swaption_price<SwaptionSide::payer>(
        parameters,
        state,
        valuation_time,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float european_payer_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
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
        valuation_time,
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
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return european_swaption_price<SwaptionSide::receiver>(
        parameters,
        state,
        valuation_time,
        exercise_time,
        fixed_rate,
        schedule
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float european_receiver_swaption_price(
    const FittedParameters<CurveProvider>& parameters,
    float state,
    float valuation_time,
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
        valuation_time,
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

// Curve-independent model adapter consumed by Bermudan pricing policies.
template<typename FittedModelComposition>
struct BermudanSwaptionAnalyticsPolicy {
    using ModelParameters =
        typename FittedModelComposition::ModelParameters;
    using CurveParameters =
        typename FittedModelComposition::CurveParameters;
    using PreparedModel = typename FittedModelComposition::FittedModel;

    struct PreparedRegressionState {
        float inverse_scale;
    };

    __device__ __forceinline__ static PreparedRegressionState
    prepare_regression_state(const ModelParameters& parameters) {
        return {
            ::ai_factory::workbench::fixed_income::
                mean_reverting_gaussian::
                    inverse_stationary_deviation_from_volatility(
                        parameters.volatility,
                        parameters.mean_reversion
                    ),
        };
    }

    __device__ __forceinline__ static float normalize_regression_state(
        const PreparedRegressionState& prepared,
        float state
    ) {
        return state * prepared.inverse_scale;
    }

    __device__ __forceinline__ static PreparedModel prepare_model(
        const ModelParameters& model,
        const CurveParameters& curve
    ) {
        return FittedModelComposition::compose(model, curve);
    }

    template<typename JointState>
    __device__ __forceinline__ static float factor_state(
        const JointState& state
    ) {
        return state.state;
    }

    __device__ __forceinline__ static float log_discount_factor(
        const PreparedModel& model,
        float state_integral,
        float time
    ) {
        return fitted::log_discount_factor(model, state_integral, time);
    }

    template<typename ScheduleView>
    __device__ __forceinline__ static float payer_swap_value(
        const PreparedModel& model,
        float state,
        float valuation_time,
        float start_time,
        float fixed_rate,
        const ScheduleView& schedule
    ) {
        return fitted::payer_swap_value(
            model,
            state,
            valuation_time,
            start_time,
            fixed_rate,
            schedule
        );
    }
};

}  // namespace ai_factory::workbench::model::fixed_income::hull_white::fitted
