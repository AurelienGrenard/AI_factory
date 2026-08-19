// Reusable CUDA analytics for the affine CIR short-rate model.
#pragma once

#include "model/fixed_income/cir/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::model::cir {

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the affine bond prefactor A(t,T).
__device__ __forceinline__ float log_A(
    const CirModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine bond prefactor A(t,T).
__device__ __forceinline__ float A(
    const CirModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine state loading B(t,T).
__device__ __forceinline__ float B(
    const CirModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return log P(valuation_time,maturity) = log A - B*r.
__device__ __forceinline__ float log_zero_coupon_bond(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
);

// Return the model zero-coupon bond P(valuation_time,maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float maturity
);

// Return the simple forward rate observed at valuation_time over [start,end].
__device__ __forceinline__ float forward_rate(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const CirModelParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
);

}  // namespace ai_factory::workbench::model::cir
