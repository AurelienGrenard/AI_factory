// Closed-form affine fixed-income analytics for the CIR model.
#pragma once

#include "model/fixed_income/cir/analytics.cuh"

#include "common/fixed_income/analytics_concepts.cuh"
#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/jamshidian.cuh"
#include "common/fixed_income/one_factor_affine.cuh"
#include "common/noncentral_chi_square.cuh"

#include <cuda_runtime.h>

#include <cfloat>

namespace ai_factory::workbench::model::fixed_income::cir {

// ==================== Model-specific implementation =======================

namespace {

using AffineBondCoefficients =
    ::ai_factory::workbench::fixed_income::OneFactorAffineBondCoefficients;

struct BondOptionContext {
    float expiry_bond;
    float noncentrality_numerator;
    float base_rate;
    float degrees_of_freedom;
};

// Compute log(A) and B together while sharing gamma and exp(-gamma*tau).
__device__ __forceinline__ AffineBondCoefficients affine_bond_coefficients(
    const ProcessParameters& process,
    float time_to_maturity
) {
    const float kappa = process.mean_reversion;
    const float sigma_squared = process.volatility * process.volatility;
    const float gamma = sqrtf(kappa * kappa + 2.0f * sigma_squared);
    // Avoid subtracting nearly equal gamma and kappa for narrow diffusions.
    const float gamma_minus_kappa =
        2.0f * sigma_squared / (gamma + kappa);
    const float one_minus_gamma_decay = -expm1f(
        -gamma * time_to_maturity
    );
    const float denominator = fmaf(
        -gamma_minus_kappa,
        one_minus_gamma_decay,
        2.0f * gamma
    );
    const float relative_denominator_increment =
        -gamma_minus_kappa * one_minus_gamma_decay / (2.0f * gamma);
    const float log_base = -log1pf(relative_denominator_increment)
        - 0.5f * gamma_minus_kappa * time_to_maturity;
    return {
        2.0f * kappa * process.long_term_mean / sigma_squared * log_base,
        2.0f * one_minus_gamma_decay / denominator,
    };
}

// Prepare all CIR exercise-measure invariants once per option strip.
__device__ __forceinline__ BondOptionContext prepare_bond_option_context(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry
) {
    const float expiry_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time, option_expiry
    );
    const float time_to_expiry = option_expiry - valuation_time;
    if (time_to_expiry <= 1.0e-7f) {
        return {
            expf(expiry_log_bond),
            0.0f,
            0.0f,
            0.0f,
        };
    }

    const ProcessParameters& process = parameters.process;
    const float sigma_squared = process.volatility * process.volatility;
    const float gamma = sqrtf(
        process.mean_reversion * process.mean_reversion
        + 2.0f * sigma_squared
    );
    const float gamma_decay = expf(-gamma * time_to_expiry);
    const float one_minus_gamma_decay = -expm1f(
        -gamma * time_to_expiry
    );
    const float rho = 2.0f * gamma * gamma_decay
        / (sigma_squared * one_minus_gamma_decay);
    const float psi = (process.mean_reversion + gamma) / sigma_squared;
    const float rho_squared_growth =
        4.0f * gamma * gamma * gamma_decay
        / (
            sigma_squared * sigma_squared
            * one_minus_gamma_decay * one_minus_gamma_decay
        );
    return {
        expf(expiry_log_bond),
        2.0f * rho_squared_growth * state,
        rho + psi,
        4.0f * process.mean_reversion * process.long_term_mean
            / sigma_squared,
    };
}

// Price one CIR bond option while reusing the exercise-measure context.
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
    const float underlying_bond = expf(underlying_log_bond);
    if (context.base_rate <= 0.0f) {
        return fmaxf(option_sign * (underlying_bond - strike), 0.0f);
    }

    const AffineBondCoefficients expiry_coefficients =
        affine_bond_coefficients(
            parameters.process, bond_maturity - option_expiry
        );
    const float critical_state = (
        expiry_coefficients.log_A - logf(strike)
    ) / expiry_coefficients.B;
    const float bond_rate = context.base_rate + expiry_coefficients.B;
    const DistributionProbabilities bond_measure =
        noncentral_chi_square_probabilities(
            context.degrees_of_freedom,
            context.noncentrality_numerator / bond_rate,
            2.0f * critical_state * bond_rate
        );
    const DistributionProbabilities expiry_measure =
        noncentral_chi_square_probabilities(
            context.degrees_of_freedom,
            context.noncentrality_numerator / context.base_rate,
            2.0f * critical_state * context.base_rate
        );

    if (option_sign > 0.0f) {
        return fmaxf(
            underlying_bond * bond_measure.cdf
                - strike * context.expiry_bond * expiry_measure.cdf,
            0.0f
        );
    }
    return fmaxf(
        strike * context.expiry_bond * expiry_measure.survival
            - underlying_bond * bond_measure.survival,
        0.0f
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

// Bind CIR A/B and non-central-chi-square inputs to common formulas.
struct AnalyticsProvider {
    __device__ __forceinline__ AffineBondCoefficients
    affine_bond_coefficients(
        const ModelParameters& parameters,
        float valuation_time,
        float maturity
    ) const {
        return cir::affine_bond_coefficients(
            parameters.process, maturity - valuation_time
        );
    }

    __device__ __forceinline__ float zero_coupon_bond(
        const ModelParameters& parameters,
        float state,
        float valuation_time,
        float maturity
    ) const {
        return ::ai_factory::workbench::fixed_income::zero_coupon_bond(
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
        return cir::prepare_bond_option_context(
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
        return cir::zero_coupon_bond_option_price(
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

static_assert(
    ::ai_factory::workbench::fixed_income::JamshidianAnalyticsProvider<
        AnalyticsProvider,
        ModelParameters,
        float
    >
);

}  // namespace

// The standalone CIR state is itself the short rate.
__device__ __forceinline__ float short_rate(
    const ModelParameters& parameters,
    float state,
    float time
) {
    static_cast<void>(parameters);
    static_cast<void>(time);
    return state;
}

// ===================== Common fixed-income analytics ======================

// Expose log(A) because bond logarithms should not round-trip through exp/log.
__device__ __forceinline__ float log_A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    ).log_A;
}

// Expose the textbook multiplicative prefactor used in P=A*exp(-B*r).
__device__ __forceinline__ float A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return expf(log_A(parameters, valuation_time, maturity));
}

// Expose the textbook state loading used in P=A*exp(-B*r).
__device__ __forceinline__ float B(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters.process, maturity - valuation_time
    ).B;
}

// Evaluate the affine bond in log space for numerical stability.
__device__ __forceinline__ float log_zero_coupon_bond(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return ::ai_factory::workbench::fixed_income::log_zero_coupon_bond(
        AnalyticsProvider{}, parameters, state, valuation_time, maturity
    );
}

// The CIR state is the short rate, so its integral is the log discount.
__device__ __forceinline__ float log_discount_factor(
    const ModelParameters& parameters,
    float state_integral,
    float time
) {
    static_cast<void>(parameters);
    static_cast<void>(time);
    return -state_integral;
}

// Exponentiate the accumulated short-rate integral only when required.
__device__ __forceinline__ float discount_factor(
    const ModelParameters& parameters,
    float state_integral,
    float time
) {
    return expf(log_discount_factor(parameters, state_integral, time));
}

// Exponentiate the conditional affine log bond only at the public boundary.
__device__ __forceinline__ float zero_coupon_bond(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return ::ai_factory::workbench::fixed_income::zero_coupon_bond(
        AnalyticsProvider{}, parameters, state, valuation_time, maturity
    );
}

// Apply the two-CDF CIR formula for a call on a zero-coupon bond.
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

// Apply the complementary-tail CIR formula for a zero-coupon bond put.
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
    float accrual_fraction
) {
    return ::ai_factory::workbench::fixed_income::forward_rate(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time,
        start_time,
        end_time,
        accrual_fraction
    );
}

template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::swap_rate(
        AnalyticsProvider{},
        parameters,
        state,
        valuation_time,
        start_time,
        schedule
    );
}

template<typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const ModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    return ::ai_factory::workbench::fixed_income::payer_swap_value(
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
    return ::ai_factory::workbench::fixed_income::jamshidian_state_boundary(
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
        parameters, exercise_time, fixed_rate,
        product::ExplicitEuropeanSwaptionScheduleView{
            payment_times, accrual_fractions, payment_count, time_day_fraction,
        }
    );
}

__device__ __forceinline__ float jamshidian_bond_strike(
    const ModelParameters& parameters,
    float exercise_time,
    float payment_time,
    float state_boundary
) {
    return ::ai_factory::workbench::fixed_income::jamshidian_bond_strike(
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
    return ::ai_factory::workbench::fixed_income::european_swaption_price<Side>(
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

}  // namespace ai_factory::workbench::model::fixed_income::cir
