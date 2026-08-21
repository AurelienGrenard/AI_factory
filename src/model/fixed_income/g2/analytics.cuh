// Closed-form fixed-income analytics for the standalone G2 model.
#pragma once

#include "model/fixed_income/g2/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::g2 {

// ======================= Model-specific analytics =========================

// Return the short rate r(t) = x(t) + y(t).
__device__ __forceinline__ float short_rate(const State& state);

// ===================== Common fixed-income analytics ======================

// Two affine state loadings in P(t,T)=A(t,T)exp(-B_x*x-B_y*y).
struct G2BondLoadings {
    float state_x;
    float state_y;
};

// Return the logarithm of the affine bond prefactor A(t,T).
__device__ __forceinline__ float log_A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine bond prefactor A(t,T).
__device__ __forceinline__ float A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return both affine state loadings B(t,T).
__device__ __forceinline__ G2BondLoadings B(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return log P = log A - B_x*x - B_y*y.
__device__ __forceinline__ float log_zero_coupon_bond(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float maturity
);

// Return minus the accumulated short-rate integral from time zero.
__device__ __forceinline__ float log_discount_factor(
    float state_integral
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    float state_integral
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate over [start_time,end_time].
__device__ __forceinline__ float forward_rate(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::uint32_t payment_count
);

}  // namespace ai_factory::workbench::model::g2
