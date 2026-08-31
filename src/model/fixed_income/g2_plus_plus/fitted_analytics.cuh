// Curve-agnostic G2++ analytics built from the standalone G2 process.
#pragma once

#include "common/fixed_income/analytics_concepts.cuh"
#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/gaussian_bond_option.cuh"
#include "common/fixed_income/mean_reverting_gaussian.cuh"
#include "model/fixed_income/g2/analytics.cuh"
#include "model/fixed_income/g2_plus_plus/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::fitted {

template<typename CurveProvider>
struct FittedParameters {
    model::fixed_income::g2::ProcessParameters process;
    typename CurveProvider::Parameters initial_curve;
};

template<typename CurveProvider>
__device__ __forceinline__ FittedParameters<CurveProvider>
compose_fitted_model(
    const ModelParameters& parameters,
    const typename CurveProvider::Parameters& initial_curve
) {
    return {parameters.process, initial_curve};
}

template<typename CurveProvider>
__device__ __forceinline__ float short_rate_shift(
    const FittedParameters<CurveProvider>& parameters,
    float time_years
) {
    const float mean_reversion_x = parameters.process.mean_reversion_x;
    const float mean_reversion_y = parameters.process.mean_reversion_y;
    const float one_minus_decay_x = -expm1f(-mean_reversion_x * time_years);
    const float one_minus_decay_y = -expm1f(-mean_reversion_y * time_years);
    const float correction_x =
        parameters.process.volatility_x * parameters.process.volatility_x
        * one_minus_decay_x * one_minus_decay_x
        / (2.0f * mean_reversion_x * mean_reversion_x);
    const float correction_y =
        parameters.process.volatility_y * parameters.process.volatility_y
        * one_minus_decay_y * one_minus_decay_y
        / (2.0f * mean_reversion_y * mean_reversion_y);
    const float cross_correction = parameters.process.correlation
        * parameters.process.volatility_x * parameters.process.volatility_y
        * one_minus_decay_x * one_minus_decay_y
        / (mean_reversion_x * mean_reversion_y);
    return CurveProvider::instantaneous_forward(
        parameters.initial_curve, time_years
    ) + correction_x + correction_y + cross_correction;
}

template<typename CurveProvider>
__device__ __forceinline__ float short_rate(
    const FittedParameters<CurveProvider>& parameters,
    const model::fixed_income::g2::State& state,
    float time_years
) {
    return model::fixed_income::g2::stochastic_short_rate(state)
        + short_rate_shift(parameters, time_years);
}

template<typename CurveProvider>
__device__ __forceinline__ float shift_integral(
    const FittedParameters<CurveProvider>& parameters,
    float start_time_years,
    float end_time_years
) {
    const float forward_integral =
        CurveProvider::log_discount_factor(
            parameters.initial_curve, start_time_years
        )
        - CurveProvider::log_discount_factor(
            parameters.initial_curve, end_time_years
        );
    const float start_variance =
        model::fixed_income::g2::integral_moments(
            parameters.process, start_time_years
        ).variance;
    const float end_variance =
        model::fixed_income::g2::integral_moments(
            parameters.process, end_time_years
        ).variance;
    return forward_integral + 0.5f * (end_variance - start_variance);
}

template<typename CurveProvider>
__device__ __forceinline__
model::fixed_income::g2::TwoFactorAffineBondCoefficients
affine_bond_coefficients(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
) {
    const auto base_coefficients =
        model::fixed_income::g2::affine_bond_coefficients(
            parameters.process, maturity_years - valuation_time_years
        );
    if (valuation_time_years == 0.0f) {
        return {
            CurveProvider::log_discount_factor(
                parameters.initial_curve, maturity_years
            ),
            base_coefficients.B,
        };
    }
    return {
        -shift_integral(parameters, valuation_time_years, maturity_years)
            + base_coefficients.log_A,
        base_coefficients.B,
    };
}

template<typename CurveProvider>
struct BondOptionContext {
    ::ai_factory::workbench::fixed_income::GaussianBondOptionDiscountContext
        discount;
};

template<typename CurveProvider>
struct AnalyticsProvider {
    using Parameters = FittedParameters<CurveProvider>;

    __device__ __forceinline__ float zero_coupon_bond(
        const Parameters& parameters,
        const model::fixed_income::g2::State& state,
        float valuation_time_years,
        float maturity_years
    ) const {
        const auto coefficients = affine_bond_coefficients(
            parameters, valuation_time_years, maturity_years
        );
        return expf(fmaf(
            -coefficients.B.state_x,
            state.state_x,
            fmaf(
                -coefficients.B.state_y,
                state.state_y,
                coefficients.log_A
            )
        ));
    }

    __device__ __forceinline__ BondOptionContext<CurveProvider>
    prepare_bond_option_context(
        const Parameters& parameters,
        const model::fixed_income::g2::State& state,
        float valuation_time_years,
        float option_expiry_years
    ) const {
        const auto coefficients = affine_bond_coefficients(
            parameters, valuation_time_years, option_expiry_years
        );
        const float expiry_log_bond = fmaf(
            -coefficients.B.state_x,
            state.state_x,
            fmaf(
                -coefficients.B.state_y,
                state.state_y,
                coefficients.log_A
            )
        );
        return {{expiry_log_bond, expf(expiry_log_bond)}};
    }

    __device__ __forceinline__ float bond_option_price(
        const BondOptionContext<CurveProvider>& context,
        const Parameters& parameters,
        const model::fixed_income::g2::State& state,
        float option_sign,
        float valuation_time_years,
        float option_expiry_years,
        float bond_maturity_years,
        float strike
    ) const {
        const auto coefficients = affine_bond_coefficients(
            parameters, valuation_time_years, bond_maturity_years
        );
        const float underlying_log_bond = fmaf(
            -coefficients.B.state_x,
            state.state_x,
            fmaf(
                -coefficients.B.state_y,
                state.state_y,
                coefficients.log_A
            )
        );
        const float total_volatility =
            model::fixed_income::g2::bond_option_total_volatility(
                parameters.process,
                option_expiry_years - valuation_time_years,
                bond_maturity_years - option_expiry_years
            );
        return ::ai_factory::workbench::fixed_income::
            discounted_lognormal_bond_option_price(
                context.discount,
                underlying_log_bond,
                total_volatility,
                strike,
                option_sign
            );
    }
};

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
__device__ __forceinline__
model::fixed_income::g2::TwoFactorAffineBondLoadings B(
    const FittedParameters<CurveProvider>& parameters,
    float valuation_time_years,
    float maturity_years
) {
    return model::fixed_income::g2::affine_bond_coefficients(
        parameters.process, maturity_years - valuation_time_years
    ).B;
}

template<typename CurveProvider>
__device__ __forceinline__ float log_zero_coupon_bond(
    const FittedParameters<CurveProvider>& parameters,
    const model::fixed_income::g2::State& state,
    float valuation_time_years,
    float maturity_years
) {
    const auto coefficients = affine_bond_coefficients(
        parameters, valuation_time_years, maturity_years
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
    const model::fixed_income::g2::State& state,
    float valuation_time_years,
    float maturity_years
) {
    return AnalyticsProvider<CurveProvider>{}.zero_coupon_bond(
        parameters, state, valuation_time_years, maturity_years
    );
}

template<typename CurveProvider>
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const FittedParameters<CurveProvider>& parameters,
    const model::fixed_income::g2::State& state,
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
    const model::fixed_income::g2::State& state,
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
    const model::fixed_income::g2::State& state,
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
    const model::fixed_income::g2::State& state,
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
    const model::fixed_income::g2::State& state,
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
    const model::fixed_income::g2::State& state,
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

// Curve-independent model adapter consumed by Bermudan pricing policies.
template<typename FittedModelComposition>
struct BermudanSwaptionAnalyticsPolicy {
    using ModelParameters =
        typename FittedModelComposition::ModelParameters;
    using CurveParameters =
        typename FittedModelComposition::CurveParameters;
    using PreparedModel = typename FittedModelComposition::FittedModel;

    struct PreparedRegressionState {
        float inverse_scale_x;
        float inverse_scale_y;
    };

    __device__ __forceinline__ static PreparedRegressionState
    prepare_regression_state(const ModelParameters& parameters) {
        return {
            ::ai_factory::workbench::fixed_income::
                mean_reverting_gaussian::
                    inverse_stationary_deviation_from_volatility(
                        parameters.process.volatility_x,
                        parameters.process.mean_reversion_x
                    ),
            ::ai_factory::workbench::fixed_income::
                mean_reverting_gaussian::
                    inverse_stationary_deviation_from_volatility(
                        parameters.process.volatility_y,
                        parameters.process.mean_reversion_y
                    ),
        };
    }

    __device__ __forceinline__ static float normalize_regression_state_x(
        const PreparedRegressionState& prepared,
        float state_x
    ) {
        return state_x * prepared.inverse_scale_x;
    }

    __device__ __forceinline__ static float normalize_regression_state_y(
        const PreparedRegressionState& prepared,
        float state_y
    ) {
        return state_y * prepared.inverse_scale_y;
    }

    __device__ __forceinline__ static PreparedModel prepare_model(
        const ModelParameters& model,
        const CurveParameters& curve
    ) {
        return FittedModelComposition::compose(model, curve);
    }

    template<typename JointState>
    __device__ __forceinline__ static model::fixed_income::g2::State
    factor_state(const JointState& state) {
        return state.state;
    }

    __device__ __forceinline__ static float log_discount_factor(
        const PreparedModel& model,
        float state_integral,
        float time_years
    ) {
        return fitted::log_discount_factor(model, state_integral, time_years);
    }

    template<typename ScheduleView>
    __device__ __forceinline__ static float payer_swap_value(
        const PreparedModel& model,
        const model::fixed_income::g2::State& state,
        float valuation_time_years,
        float start_time_years,
        float fixed_rate,
        const ScheduleView& schedule
    ) {
        return fitted::payer_swap_value(
            model,
            state,
            valuation_time_years,
            start_time_years,
            fixed_rate,
            schedule
        );
    }
};

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::fitted
