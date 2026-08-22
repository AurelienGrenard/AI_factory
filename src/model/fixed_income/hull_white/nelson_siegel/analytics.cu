// Hull-White analytics composed from OU and Nelson-Siegel formulas.
#include "model/fixed_income/hull_white/nelson_siegel/analytics.cuh"

#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/gaussian_bond_option.cuh"
#include "common/fixed_income/jamshidian.cuh"
#include "common/fixed_income/one_factor_affine.cuh"

// Include implementations so NVCC can inline process and curve formulas.
#include "curve/nelson_siegel/term_structure.cu"
#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cu"

#include <cuda_runtime.h>

#include <cfloat>

namespace ai_factory::workbench::model::hull_white::nelson_siegel {

// ======================= Model-specific analytics =========================

// Compose the curve-independent OU parameters with the fitted initial curve.
__device__ __forceinline__ HullWhiteFittedParameters compose_model(
    const ModelParameters& parameters,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve
) {
    return {
        {parameters.mean_reversion, parameters.volatility},
        initial_curve,
    };
}

// Evaluate the deterministic shift that fits the initial forward curve.
__device__ __forceinline__ float short_rate_shift(
    const HullWhiteFittedParameters& parameters,
    float time
) {
    const float a = parameters.process.mean_reversion;
    const float one_minus_decay = -expm1f(-a * time);
    const float correction =
        parameters.process.volatility * parameters.process.volatility
        * one_minus_decay * one_minus_decay / (2.0f * a * a);
    return curve::nelson_siegel::instantaneous_forward(
        parameters.initial_curve, time
    ) + correction;
}

// Add the Gaussian state to the fitted deterministic curve shift.
__device__ __forceinline__ float short_rate(
    const HullWhiteFittedParameters& parameters,
    float state,
    float time
) {
    return state + short_rate_shift(parameters, time);
}

namespace {

// Integrate the Nelson-Siegel forward curve and Hull-White convexity shift.
__device__ __forceinline__ float shift_integral(
    const HullWhiteFittedParameters& parameters,
    float start,
    float end
) {
    const float a = parameters.process.mean_reversion;
    const float delta = end - start;
    const float one_minus_decay = -expm1f(-a * delta);
    const float one_minus_decay_squared = -expm1f(-2.0f * a * delta);
    const float forward_integral =
        curve::nelson_siegel::log_discount_factor(
            parameters.initial_curve, start
        )
        - curve::nelson_siegel::log_discount_factor(
            parameters.initial_curve, end
        );
    if (start == 0.0f) {
        return forward_integral
            + 0.5f
                * model::ornstein_uhlenbeck::integral_variance(
                    parameters.process, end
                );
    }
    const float convexity_integral =
        parameters.process.volatility * parameters.process.volatility
        / (2.0f * a * a)
        * (
            delta
            - 2.0f * expf(-a * start) * one_minus_decay / a
            + expf(-2.0f * a * start)
                * one_minus_decay_squared / (2.0f * a)
        );
    return forward_integral + convexity_integral;
}

// Reuse the common one-factor affine and Gaussian option shells.
using AffineBondCoefficients =
    fixed_income::OneFactorAffineBondCoefficients;

struct BondOptionContext {
    fixed_income::GaussianBondOptionDiscountContext discount;
    float expiry_state_standard_deviation;
};

// Compute fitted log(A) and B from one shared OU-moment evaluation.
__device__ __forceinline__ AffineBondCoefficients affine_bond_coefficients(
    const HullWhiteFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    const float delta = maturity - valuation_time;
    const model::ornstein_uhlenbeck::IntegralMoments moments =
        model::ornstein_uhlenbeck::integral_moments(
            parameters.process, delta
        );
    // At t=0, use the fitted curve directly and avoid cancelling shift terms.
    if (valuation_time == 0.0f) {
        return {
            curve::nelson_siegel::log_discount_factor(
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

// Prepare all model/expiry invariants once for a strip of bond options.
__device__ __forceinline__ BondOptionContext prepare_bond_option_context(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry
) {
    const float expiry_log_bond = log_zero_coupon_bond(
        parameters, state, valuation_time, option_expiry
    );
    const float a = parameters.process.mean_reversion;
    const float time_to_expiry = option_expiry - valuation_time;
    const float state_variance =
        parameters.process.volatility * parameters.process.volatility
        * (-expm1f(-2.0f * a * time_to_expiry)) / (2.0f * a);
    return {
        {expiry_log_bond, expf(expiry_log_bond)},
        sqrtf(fmaxf(state_variance, 0.0f)),
    };
}

// Price one bond option while reusing the prepared expiry context.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const BondOptionContext& context,
    const HullWhiteFittedParameters& parameters,
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
    const float bond_loading =
        model::ornstein_uhlenbeck::integral_state_loading(
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
    const HullWhiteFittedParameters& parameters,
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

// Bind fitted A/B and Gaussian option inputs to common formulas.
struct AnalyticsProvider {
    __device__ __forceinline__ AffineBondCoefficients
    affine_bond_coefficients(
        const HullWhiteFittedParameters& parameters,
        float valuation_time,
        float maturity
    ) const {
        return ai_factory::workbench::model::hull_white::nelson_siegel::affine_bond_coefficients(
            parameters, valuation_time, maturity
        );
    }

    __device__ __forceinline__ float zero_coupon_bond(
        const HullWhiteFittedParameters& parameters,
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
        const HullWhiteFittedParameters& parameters,
        float state,
        float valuation_time,
        float option_expiry
    ) const {
        return ai_factory::workbench::model::hull_white::nelson_siegel::prepare_bond_option_context(
            parameters, state, valuation_time, option_expiry
        );
    }

    __device__ __forceinline__ float bond_option_price(
        const BondOptionContext& context,
        const HullWhiteFittedParameters& parameters,
        float state,
        float option_sign,
        float valuation_time,
        float option_expiry,
        float bond_maturity,
        float strike
    ) const {
        return ai_factory::workbench::model::hull_white::nelson_siegel::zero_coupon_bond_option_price(
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

// Return the logarithm of the fitted affine prefactor.
__device__ __forceinline__ float log_A(
    const HullWhiteFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    return affine_bond_coefficients(
        parameters, valuation_time, maturity
    ).log_A;
}

// Exponentiate the affine prefactor only for callers requesting A itself.
__device__ __forceinline__ float A(
    const HullWhiteFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    return expf(log_A(parameters, valuation_time, maturity));
}

// Return the OU-factor loading in the fitted affine expression.
__device__ __forceinline__ float B(
    const HullWhiteFittedParameters& parameters,
    float valuation_time,
    float maturity
) {
    return model::ornstein_uhlenbeck::integral_state_loading(
        parameters.process.mean_reversion, maturity - valuation_time
    );
}

// Evaluate log(A)-B*x from one grouped coefficient calculation.
__device__ __forceinline__ float log_zero_coupon_bond(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return fixed_income::log_zero_coupon_bond(
        AnalyticsProvider{}, parameters, state, valuation_time, maturity
    );
}

// Combine the stochastic integral with the analytical curve shift.
__device__ __forceinline__ float log_discount_factor(
    const HullWhiteFittedParameters& parameters,
    float state_integral,
    float time
) {
    return -state_integral
        - shift_integral(parameters, 0.0f, time);
}

// Exponentiate the exact path log-discount only when a payoff needs it.
__device__ __forceinline__ float discount_factor(
    const HullWhiteFittedParameters& parameters,
    float state_integral,
    float time
) {
    return expf(log_discount_factor(parameters, state_integral, time));
}

// Price one zero-coupon from the conditional Gaussian state integral.
__device__ __forceinline__ float zero_coupon_bond(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return expf(
        log_zero_coupon_bond(parameters, state, valuation_time, maturity)
    );
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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

template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const HullWhiteFittedParameters& parameters,
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

template<typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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
    const HullWhiteFittedParameters& parameters,
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

}  // namespace ai_factory::workbench::model::hull_white::nelson_siegel
