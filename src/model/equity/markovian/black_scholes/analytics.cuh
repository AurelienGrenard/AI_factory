// Public closed-form analytics declarations used by Black-Scholes pricing bindings.
#pragma once

#include "common/lognormal_option.cuh"
#include "model/equity/markovian/black_scholes/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::black_scholes {

struct BlackScholesAnalyticsContext {
    float log_spot;
    float risk_free_rate;
    float dividend_yield;
    float volatility;
    float variance;
};

struct LognormalEvolutionContext {
    float log_spot;
    float log_drift_rate;
    float volatility;
};

__device__ __forceinline__ BlackScholesAnalyticsContext prepare_analytics(
    const ModelParameters& parameters
);

__device__ __forceinline__ DiscountedLognormalOptionContext
prepare_option_context(
    const BlackScholesAnalyticsContext& analytics,
    float maturity_years
);

__device__ __forceinline__ LognormalEvolutionContext
prepare_lognormal_evolution(
    const BlackScholesAnalyticsContext& analytics
);

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_vanilla_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float strike,
    float maturity_years
);

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_gap_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float trigger_strike,
    float payoff_strike,
    float maturity_years
);

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_forward_start_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float moneyness,
    float reset_time_years,
    float maturity_years
);

__device__ __forceinline__ DiscountedLognormalOptionValues
prepare_geometric_asian_option_values(
    const BlackScholesAnalyticsContext& analytics,
    float strike,
    float maturity_years,
    std::uint32_t transition_count
);

__device__ __forceinline__ float cash_or_nothing_price(
    float discounted_cash_payoff,
    float d2,
    float option_sign
);

__device__ __forceinline__ float asset_or_nothing_price(
    float discounted_spot,
    float d1,
    float option_sign
);

__device__ __forceinline__ float lognormal_interval_probability(
    const BlackScholesAnalyticsContext& analytics,
    float lower_level,
    float upper_level,
    float observation_time_years
);

__device__ __forceinline__ float lognormal_log_interval_probability(
    const LognormalEvolutionContext& evolution,
    float log_lower_level,
    float log_upper_level,
    float observation_time_years
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
