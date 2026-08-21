// Hull-White analytics composed from OU and Nelson-Siegel formulas.
#include "model/fixed_income/hull_white/nelson_siegel/analytics.cuh"

// Include implementations so NVCC can inline process and curve formulas.
#include "curve/nelson_siegel/term_structure.cu"
#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cu"

#include <cuda_runtime.h>

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

// The helpers below support the same analytical interface as standalone OU.
constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

// Evaluate the standard normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    return 0.5f * erfcf(-value * kInverseSqrtTwo);
}

struct AffineBondCoefficients {
    float log_A;
    float B;
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

// Price a call (+1) or put (-1) with one shared Black-style expression.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const HullWhiteFittedParameters& parameters,
    float state,
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

    const float a = parameters.process.mean_reversion;
    const float time_to_expiry = option_expiry - valuation_time;
    const float state_variance =
        parameters.process.volatility * parameters.process.volatility
        * (-expm1f(-2.0f * a * time_to_expiry)) / (2.0f * a);
    const float bond_loading =
        model::ornstein_uhlenbeck::integral_state_loading(
            a, bond_maturity - option_expiry
        );
    const float total_volatility =
        bond_loading * sqrtf(fmaxf(state_variance, 0.0f));
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
                - strike * expiry_bond
                    * normal_cdf(option_sign * d2)
            ),
        0.0f
    );
}

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
    const AffineBondCoefficients coefficients = affine_bond_coefficients(
        parameters, valuation_time, maturity
    );
    return fmaf(-coefficients.B, state, coefficients.log_A);
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
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::uint32_t payment_count
) {
    float annuity = 0.0f;
    float end_bond = 0.0f;
    for (std::uint32_t payment = 0U;
         payment < payment_count;
         ++payment) {
        const float current_bond = zero_coupon_bond(
            parameters,
            state,
            valuation_time,
            payment_times[payment]
        );
        annuity = fmaf(
            accrual_periods[payment],
            current_bond,
            annuity
        );
        end_bond = current_bond;
    }
    const float start_bond = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    // Value the floating leg from its start and final zero-coupon bonds.
    return (start_bond - end_bond) / annuity;
}

}  // namespace ai_factory::workbench::model::hull_white::nelson_siegel
