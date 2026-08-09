// Reusable CUDA analytics for the Ornstein-Uhlenbeck process.
#pragma once

#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Return the accumulated path log-discount from time zero.
__device__ __forceinline__ float log_discount_factor(
    const joint::OrnsteinUhlenbeckJointState& joint_state
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    const joint::OrnsteinUhlenbeckJointState& joint_state
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const OrnsteinUhlenbeckModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const OrnsteinUhlenbeckModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const OrnsteinUhlenbeckModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate observed at valuation_time over [start,end].
__device__ __forceinline__ float forward_rate(
    const OrnsteinUhlenbeckModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const OrnsteinUhlenbeckModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
);

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
