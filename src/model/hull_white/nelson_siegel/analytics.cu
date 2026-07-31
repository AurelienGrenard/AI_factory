// Hull-White analytics composed from OU and Nelson-Siegel formulas.
#include "model/hull_white/nelson_siegel/analytics.cuh"

// Include implementations so NVCC can inline process and curve formulas.
#include "curve/nelson_siegel/term_structure.cu"
#include "model/ornstein_uhlenbeck/analytics.cu"

#include <cuda_runtime.h>

namespace ai_factory::workbench::hull_white::nelson_siegel {
namespace {

constexpr float kInverseSqrtTwo = 0.70710678118654752440f;

// Evaluate the standard normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    return 0.5f * erfcf(-value * kInverseSqrtTwo);
}

// Integrate the deterministic shift phi between two model times.
__device__ __forceinline__ float shift_integral(
    const HullWhiteParameters& parameters,
    float start,
    float end
) {
    const float a = parameters.process.mean_reversion;
    const float delta = end - start;
    const float one_minus_decay = -expm1f(-a * delta);
    const float one_minus_decay_squared = -expm1f(-2.0f * a * delta);
    const float forward_integral =
        curve::nelson_siegel::log_discount(parameters.initial_curve, start)
        - curve::nelson_siegel::log_discount(parameters.initial_curve, end);
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

// Return the total log-volatility of a zero-coupon at option expiry.
__device__ __forceinline__ float zero_coupon_bond_option_volatility(
    const HullWhiteParameters& parameters,
    float valuation_time,
    float option_expiry,
    float bond_maturity
) {
    const float a = parameters.process.mean_reversion;
    const float time_to_expiry = option_expiry - valuation_time;
    const float factor_variance =
        parameters.process.volatility * parameters.process.volatility
        * (-expm1f(-2.0f * a * time_to_expiry)) / (2.0f * a);
    const float bond_loading =
        model::ornstein_uhlenbeck::integral_factor_loading(
            a, bond_maturity - option_expiry
        );
    return bond_loading * sqrtf(fmaxf(factor_variance, 0.0f));
}

// Price a call (+1) or put (-1) with one shared Black-style expression.
__device__ __forceinline__ float zero_coupon_bond_option_price(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float option_sign,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
) {
    const float expiry_discount = zero_coupon_bond(
        parameters, state, valuation_time, option_expiry
    );
    const float bond_discount = zero_coupon_bond(
        parameters, state, valuation_time, bond_maturity
    );
    const float volatility = zero_coupon_bond_option_volatility(
        parameters, valuation_time, option_expiry, bond_maturity
    );
    if (volatility <= 1.0e-7f) {
        return fmaxf(
            option_sign * (bond_discount - strike * expiry_discount),
            0.0f
        );
    }

    const float d1 =
        logf(bond_discount / (strike * expiry_discount)) / volatility
        + 0.5f * volatility;
    const float d2 = d1 - volatility;
    return fmaxf(
        option_sign
            * (
                bond_discount * normal_cdf(option_sign * d1)
                - strike * expiry_discount
                    * normal_cdf(option_sign * d2)
            ),
        0.0f
    );
}

}  // namespace

// Compose the curve-independent OU parameters with the fitted initial curve.
__device__ __forceinline__ HullWhiteParameters prepare_model(
    const HullWhiteModelParameters& parameters,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve
) {
    return {
        {parameters.mean_reversion, parameters.volatility},
        initial_curve,
    };
}

// Evaluate the deterministic shift that fits the initial forward curve.
__device__ __forceinline__ float short_rate_shift(
    const HullWhiteParameters& parameters,
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

// Add the Gaussian factor to the fitted deterministic curve shift.
__device__ __forceinline__ float short_rate(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float time
) {
    return state.factor + short_rate_shift(parameters, time);
}

// Combine the stochastic integral with the analytical curve shift.
__device__ __forceinline__ float log_discount(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float time
) {
    return -state.integrated_factor
        - shift_integral(parameters, 0.0f, time);
}

// Exponentiate the exact path log-discount only when a payoff needs it.
__device__ __forceinline__ float discount_factor(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float time
) {
    return expf(log_discount(parameters, state, time));
}

// Price one zero-coupon from the conditional Gaussian factor integral.
__device__ __forceinline__ float zero_coupon_bond(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float maturity
) {
    const float delta = maturity - valuation_time;
    const float loading =
        model::ornstein_uhlenbeck::integral_factor_loading(
            parameters.process.mean_reversion, delta
        );
    const float deterministic_integral =
        shift_integral(parameters, valuation_time, maturity);
    const float variance =
        model::ornstein_uhlenbeck::integral_variance(
            parameters.process, delta
        );
    return expf(
        -loading * state.factor
        - deterministic_integral
        + 0.5f * variance
    );
}

// Apply the closed-form call formula to the conditional bond forward.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
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
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
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
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
) {
    const float start_discount = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    const float end_discount = zero_coupon_bond(
        parameters, state, valuation_time, end_time
    );
    return (start_discount / end_discount - 1.0f) / accrual_period;
}

// Divide the conditional floating-leg value by the fixed-leg annuity.
__device__ __forceinline__ float swap_rate(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
) {
    float annuity = 0.0f;
    for (std::size_t payment = 0U;
         payment < payment_count;
         ++payment) {
        annuity = fmaf(
            accrual_periods[payment],
            zero_coupon_bond(
                parameters,
                state,
                valuation_time,
                payment_times[payment]
            ),
            annuity
        );
    }
    const float start_discount = zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    const float end_discount = zero_coupon_bond(
        parameters,
        state,
        valuation_time,
        payment_times[payment_count - 1U]
    );
    return (start_discount - end_discount) / annuity;
}

}  // namespace ai_factory::workbench::hull_white::nelson_siegel
