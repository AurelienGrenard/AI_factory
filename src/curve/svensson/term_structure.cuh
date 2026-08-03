// Reusable CUDA interface for the analytical Svensson curve.
#pragma once

#include "curve/svensson/dataset.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::curve::svensson {

// Return the continuously compounded zero rate z(0, maturity).
__device__ __forceinline__ float zero_rate(
    const SvenssonParameters& parameters,
    float maturity
);

// Return log P(0, maturity) without exponentiating the discount factor.
__device__ __forceinline__ float log_discount_factor(
    const SvenssonParameters& parameters,
    float maturity
);

// Return the discount factor P(0, maturity).
__device__ __forceinline__ float discount_factor(
    const SvenssonParameters& parameters,
    float maturity
);

// Return the instantaneous forward rate f(0, maturity).
__device__ __forceinline__ float instantaneous_forward(
    const SvenssonParameters& parameters,
    float maturity
);

// Return the maturity derivative of the instantaneous forward rate.
__device__ __forceinline__ float forward_derivative(
    const SvenssonParameters& parameters,
    float maturity
);

// Return the continuously compounded forward rate over [start, end].
__device__ __forceinline__ float forward_rate(
    const SvenssonParameters& parameters,
    float start,
    float end
);

}  // namespace ai_factory::workbench::curve::svensson
