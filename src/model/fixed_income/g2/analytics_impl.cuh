// Included device definitions of the standalone G2 closed-form analytics contract.
#pragma once

#include "model/fixed_income/g2/analytics.cuh"

#include "common/fixed_income/analytics_concepts.cuh"
#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/gaussian_bond_option.cuh"

// Include exact G2 moments used by the analytical formulas below.
#include "model/fixed_income/g2/dynamics_impl.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::g2 {

// ======================= Model-specific analytics =========================

__device__ __forceinline__ float stochastic_short_rate(const State& state) {
    return state.state_x + state.state_y;
}

// Add both Gaussian factor states to reconstruct the short rate.
__device__ __forceinline__ float short_rate(
    const ModelParameters& parameters,
    const State& state,
    float time_years
) {
    static_cast<void>(parameters);
    static_cast<void>(time_years);
    return stochastic_short_rate(state);
}

// ==================== Model-specific implementation =======================

// Compute log(A), B_x, and B_y from one shared integral-moment evaluation.
__device__ __forceinline__ TwoFactorAffineBondCoefficients
affine_bond_coefficients(
    const ProcessParameters& parameters,
    float time_to_maturity
) {
    const IntegralMoments moments = integral_moments(
        parameters, time_to_maturity
    );
    return {
        0.5f * moments.variance,
        {moments.state_x_loading, moments.state_y_loading},
    };
}

// Return the conditional covariance matrix of both future factor states.
__device__ __forceinline__ void state_covariances(
    const ProcessParameters& parameters,
    float time_to_expiry,
    float& variance_x,
    float& variance_y,
    float& covariance_xy
) {
    const float a = parameters.mean_reversion_x;
    const float b = parameters.mean_reversion_y;
    variance_x = parameters.volatility_x * parameters.volatility_x
        * (-expm1f(-2.0f * a * time_to_expiry)) / (2.0f * a);
    variance_y = parameters.volatility_y * parameters.volatility_y
        * (-expm1f(-2.0f * b * time_to_expiry)) / (2.0f * b);
    covariance_xy = parameters.correlation
        * parameters.volatility_x * parameters.volatility_y
        * (-expm1f(-(a + b) * time_to_expiry)) / (a + b);
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

namespace {

using BondOptionContext =
    ::ai_factory::workbench::fixed_income::GaussianBondOptionDiscountContext;

__device__ __forceinline__ BondOptionContext prepare_bond_option_context(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time_years,
    float option_expiry_years
) {
    const float expiry_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time_years, option_expiry_years
    );
    return {expiry_log_bond, expf(expiry_log_bond)};
}

// Price a call (+1) or put (-1) with one shared Black-style expression.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const BondOptionContext& context,
    const ModelParameters& parameters,
    const State& state,
    float option_sign,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
) {
    const float underlying_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time_years, bond_maturity_years
    );
    const float total_volatility = bond_option_total_volatility(
        parameters.process,
        option_expiry_years - valuation_time_years,
        bond_maturity_years - option_expiry_years
    );
    return ::ai_factory::workbench::fixed_income::discounted_lognormal_bond_option_price(
        context,
        underlying_log_bond,
        total_volatility,
        strike,
        option_sign
    );
}

__device__ __forceinline__ float zero_coupon_bond_option_price(
    const ModelParameters& parameters,
    const State& state,
    float option_sign,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
) {
    return zero_coupon_bond_option_price(
        prepare_bond_option_context(
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

// Bind the two-factor bond function to common cashflow formulas.
struct AnalyticsProvider {
    __device__ __forceinline__ float zero_coupon_bond(
        const ModelParameters& parameters,
        const State& state,
        float valuation_time_years,
        float maturity_years
    ) const {
        return g2::zero_coupon_bond(
            parameters, state, valuation_time_years, maturity_years
        );
    }

    __device__ __forceinline__ BondOptionContext
    prepare_bond_option_context(
        const ModelParameters& parameters,
        const State& state,
        float valuation_time_years,
        float option_expiry_years
    ) const {
        return g2::prepare_bond_option_context(
            parameters, state, valuation_time_years, option_expiry_years
        );
    }

    __device__ __forceinline__ float bond_option_price(
        const BondOptionContext& context,
        const ModelParameters& parameters,
        const State& state,
        float option_sign,
        float valuation_time_years,
        float option_expiry_years,
        float bond_maturity_years,
        float strike
    ) const {
        return g2::zero_coupon_bond_option_price(
            context,
            parameters,
            state,
            option_sign,
            valuation_time_years,
            option_expiry_years,
            bond_maturity_years,
            strike
        );
    }
};

static_assert(
    ::ai_factory::workbench::fixed_income::ZeroCouponBondProvider<
        AnalyticsProvider,
        ModelParameters,
        State
    >
);
static_assert(
    ::ai_factory::workbench::fixed_income::BondOptionProvider<
        AnalyticsProvider,
        ModelParameters,
        State
    >
);

}  // namespace

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the multiplicative affine prefactor.
__device__ __forceinline__ float log_A(
    const ModelParameters& parameters,
    float valuation_time_years,
    float maturity_years
) {
    return affine_bond_coefficients(
        parameters.process, maturity_years - valuation_time_years
    ).log_A;
}

// Exponentiate the affine prefactor only for callers requesting A itself.
__device__ __forceinline__ float A(
    const ModelParameters& parameters,
    float valuation_time_years,
    float maturity_years
) {
    return expf(log_A(parameters, valuation_time_years, maturity_years));
}

// Return both factor loadings in one value.
__device__ __forceinline__ TwoFactorAffineBondLoadings B(
    const ModelParameters& parameters,
    float valuation_time_years,
    float maturity_years
) {
    const float delta = maturity_years - valuation_time_years;
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
    float valuation_time_years,
    float maturity_years
) {
    const TwoFactorAffineBondCoefficients coefficients = affine_bond_coefficients(
        parameters.process, maturity_years - valuation_time_years
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
    const ModelParameters& parameters,
    float state_integral,
    float time_years
) {
    static_cast<void>(parameters);
    static_cast<void>(time_years);
    return -state_integral;
}

// Exponentiate the accumulated rate integral only when required.
__device__ __forceinline__ float discount_factor(
    const ModelParameters& parameters,
    float state_integral,
    float time_years
) {
    return expf(log_discount_factor(parameters, state_integral, time_years));
}

// Price one zero-coupon from the conditional Gaussian rate integral.
__device__ __forceinline__ float zero_coupon_bond(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time_years,
    float maturity_years
) {
    return expf(log_zero_coupon_bond(
        parameters, state, valuation_time_years, maturity_years
    ));
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const ModelParameters& parameters,
    const State& state,
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

// Apply the closed-form put formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const ModelParameters& parameters,
    const State& state,
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

// Build one simple forward rate from two conditional zero-coupons.
__device__ __forceinline__ float forward_rate(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time_years,
    float start_time_years,
    float end_time_years,
    float accrual_fraction
) {
    return ::ai_factory::workbench::fixed_income::forward_rate(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time_years,
        start_time_years,
        end_time_years,
        accrual_fraction
    );
}

template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time_years,
    float start_time_years,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::swap_rate(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time_years,
        start_time_years,
        schedule
    );
}

template<typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time_years,
    float start_time_years,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::payer_swap_value(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time_years,
        start_time_years,
        fixed_rate,
        schedule
    );
}

}  // namespace ai_factory::workbench::model::fixed_income::g2
