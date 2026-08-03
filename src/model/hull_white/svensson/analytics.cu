// Hull-White analytics composed from OU and Svensson formulas.
#include "model/hull_white/svensson/analytics.cuh"

// Include implementations so NVCC can inline process and curve formulas.
#include "curve/svensson/term_structure.cu"
#include "model/ornstein_uhlenbeck/dynamics.cu"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::hull_white::svensson {
// Compose the curve-independent OU parameters with the fitted initial curve.
__device__ __forceinline__ HullWhiteFittedParameters compose_model(
    const HullWhiteModelParameters& parameters,
    const curve::svensson::SvenssonParameters& initial_curve
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
    return curve::svensson::instantaneous_forward(
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

// Integrate the Svensson forward curve and Hull-White convexity shift.
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
        curve::svensson::log_discount_factor(
            parameters.initial_curve, start
        )
        - curve::svensson::log_discount_factor(
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

// Return the conditional log price of one fitted zero-coupon bond.
__device__ __forceinline__ float log_zero_coupon_bond(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    // The fitted curve gives P(0,T) directly and avoids cancelling OU terms.
    if (valuation_time == 0.0f && state == 0.0f) {
        return curve::svensson::log_discount_factor(
            parameters.initial_curve, maturity
        );
    }

    const float delta = maturity - valuation_time;
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckIntegralMoments moments =
        model::ornstein_uhlenbeck::integral_moments(
            parameters.process, delta
        );
    return -moments.state_loading * state
        - shift_integral(parameters, valuation_time, maturity)
        + 0.5f * moments.variance;
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

// From here, public analytics follow the standalone OU order and names.

// Combine the stochastic integral with the analytical curve shift.
__device__ __forceinline__ float log_discount_factor(
    const HullWhiteFittedParameters& parameters,
    const model::ornstein_uhlenbeck::joint::OrnsteinUhlenbeckJointState&
        joint_state,
    float time
) {
    return -joint_state.state_integral
        - shift_integral(parameters, 0.0f, time);
}

// Exponentiate the exact path log-discount only when a payoff needs it.
__device__ __forceinline__ float discount_factor(
    const HullWhiteFittedParameters& parameters,
    const model::ornstein_uhlenbeck::joint::OrnsteinUhlenbeckJointState&
        joint_state,
    float time
) {
    return expf(log_discount_factor(parameters, joint_state, time));
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
    std::size_t payment_count
) {
    float annuity = 0.0f;
    float end_bond = 0.0f;
    for (std::size_t payment = 0U;
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

}  // namespace ai_factory::workbench::model::hull_white::svensson
