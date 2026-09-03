// Reusable closed-form analytics for the Black-Scholes model.
#pragma once

#include "model/equity/markovian/black_scholes/analytics.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::equity::black_scholes {

__device__ __forceinline__ BlackScholesAnalyticsContext prepare_analytics(
    const ModelParameters& parameters
) {
    return {
        logf(parameters.spot),
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.volatility,
        parameters.volatility * parameters.volatility,
    };
}

__device__ __forceinline__ DiscountedLognormalOptionContext
prepare_option_context(
    const BlackScholesAnalyticsContext& analytics,
    float maturity_years
) {
    const float strike_discount_log_level =
        -analytics.risk_free_rate * maturity_years;
    return {
        analytics.log_spot
            - analytics.dividend_yield * maturity_years,
        strike_discount_log_level,
        expf(strike_discount_log_level),
    };
}

__device__ __forceinline__ LognormalEvolutionContext
prepare_lognormal_evolution(
    const BlackScholesAnalyticsContext& analytics
) {
    return {
        analytics.log_spot,
        analytics.risk_free_rate - analytics.dividend_yield
            - 0.5f * analytics.variance,
        analytics.volatility,
    };
}

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_vanilla_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float strike,
    float maturity_years
) {
    return prepare_discounted_lognormal_option_values(
        prepare_option_context(analytics, maturity_years),
        analytics.volatility * sqrtf(maturity_years),
        strike
    );
}

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_gap_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float trigger_strike,
    float payoff_strike,
    float maturity_years
) {
    DiscountedLognormalOptionValues values =
        prepare_vanilla_option_values(
            analytics, trigger_strike, maturity_years
        );
    values.discounted_strike = payoff_strike
        * expf(-analytics.risk_free_rate * maturity_years);
    return values;
}

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_forward_start_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float moneyness,
    float reset_time_years,
    float maturity_years
) {
    const float remaining_time_years =
        maturity_years - reset_time_years;
    const float discounted_spot = expf(
        analytics.log_spot
        - analytics.dividend_yield * maturity_years
    );
    const float discounted_strike = moneyness * expf(
        analytics.log_spot
        - analytics.dividend_yield * reset_time_years
        - analytics.risk_free_rate * remaining_time_years
    );
    const float total_volatility =
        analytics.volatility * sqrtf(remaining_time_years);
    const float log_forward_moneyness = -logf(moneyness)
        + (analytics.risk_free_rate - analytics.dividend_yield)
            * remaining_time_years;
    float d1 = 0.0f;
    float d2 = 0.0f;
    if (total_volatility <= 1.0e-7f) {
        d1 = log_forward_moneyness > 0.0f
            ? fp32_infinity()
            : (log_forward_moneyness < 0.0f
                ? -fp32_infinity()
                : 0.0f);
        d2 = d1;
    } else {
        d1 = log_forward_moneyness / total_volatility
            + 0.5f * total_volatility;
        d2 = d1 - total_volatility;
    }
    return {discounted_spot, discounted_strike, d1, d2};
}

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_geometric_asian_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float strike,
    float maturity_years,
    std::uint32_t transition_count
) {
    const float transition_count_level =
        static_cast<float>(transition_count);
    const float log_mean = analytics.log_spot
        + 0.5f * (
            analytics.risk_free_rate - analytics.dividend_yield
            - 0.5f * analytics.variance
        ) * maturity_years;
    const float log_variance = analytics.variance * maturity_years
        * (2.0f * transition_count_level + 1.0f)
        / (6.0f * (transition_count_level + 1.0f));
    const float discount_log_level =
        -analytics.risk_free_rate * maturity_years;
    return prepare_discounted_lognormal_option_values(
        {
            discount_log_level + log_mean + 0.5f * log_variance,
            discount_log_level,
            expf(discount_log_level),
        },
        sqrtf(log_variance),
        strike
    );
}

__device__ __forceinline__ float cash_or_nothing_price(
    float discounted_cash_payoff,
    float d2,
    float option_sign
) {
    return discounted_cash_payoff * normal_cdf(option_sign * d2);
}

__device__ __forceinline__ float asset_or_nothing_price(
    float discounted_spot,
    float d1,
    float option_sign
) {
    return discounted_spot * normal_cdf(option_sign * d1);
}

__device__ __forceinline__ float lognormal_interval_probability(
    const BlackScholesAnalyticsContext& analytics,
    float lower_level,
    float upper_level,
    float observation_time_years
) {
    return lognormal_log_interval_probability(
        prepare_lognormal_evolution(analytics),
        logf(lower_level),
        logf(upper_level),
        observation_time_years
    );
}

__device__ __forceinline__ float lognormal_log_interval_probability(
    const LognormalEvolutionContext& evolution,
    float log_lower_level,
    float log_upper_level,
    float observation_time_years
) {
    const float mean = evolution.log_spot
        + evolution.log_drift_rate * observation_time_years;
    const float standard_deviation =
        evolution.volatility * sqrtf(observation_time_years);
    if (standard_deviation <= 1.0e-7f) {
        return mean > log_lower_level
            && mean < log_upper_level
            ? 1.0f
            : 0.0f;
    }
    const float lower =
        (log_lower_level - mean) / standard_deviation;
    const float upper =
        (log_upper_level - mean) / standard_deviation;
    return normal_cdf(upper) - normal_cdf(lower);
}

}  // namespace ai_factory::workbench::model::equity::black_scholes
