// Reusable CUDA analytics for the Ornstein-Uhlenbeck process.
#pragma once

#include "model/ornstein_uhlenbeck/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Return B(delta) = (1 - exp(-a delta)) / a.
__device__ __forceinline__ float integral_factor_loading(
    float mean_reversion,
    float delta
);

// Return the variance of the future integral of the Gaussian factor.
__device__ __forceinline__ float integral_variance(
    const OrnsteinUhlenbeckDynamicsParameters& parameters,
    float delta
);

// Return the current short rate carried by the OU state.
__device__ __forceinline__ float short_rate(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float time
);

// Return the accumulated path log-discount from time zero.
__device__ __forceinline__ float log_discount(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float time
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float time
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate observed at valuation_time over [start,end].
__device__ __forceinline__ float forward_rate(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const OrnsteinUhlenbeckModelParameters& parameters,
    const OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
);

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
