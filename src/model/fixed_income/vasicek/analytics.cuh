// Reusable CUDA analytics for the affine Vasicek short-rate model.
#pragma once

#include "model/fixed_income/vasicek/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::model::vasicek {

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the affine bond prefactor A(t,T).
__device__ __forceinline__ float log_A(
    const VasicekModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine bond prefactor A(t,T).
__device__ __forceinline__ float A(
    const VasicekModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine state loading B(t,T).
__device__ __forceinline__ float B(
    const VasicekModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return log P(valuation_time,maturity) = log A - B*state.
__device__ __forceinline__ float log_zero_coupon_bond(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
);

// Return the accumulated path log-discount from time zero.
__device__ __forceinline__ float log_discount_factor(
    float state_integral
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    float state_integral
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate observed at valuation_time over [start,end].
__device__ __forceinline__ float forward_rate(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const VasicekModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
);

}  // namespace ai_factory::workbench::model::vasicek
