// CUDA implementation of a flat continuously compounded curve.
#pragma once

#include "curve/flat/term_structure.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench::curve::flat {

__device__ __forceinline__ float zero_rate(
    const FlatCurveParameters& parameters,
    float
) {
    return parameters.rate;
}

__device__ __forceinline__ float log_discount_factor(
    const FlatCurveParameters& parameters,
    float maturity_years
) {
    return -parameters.rate * maturity_years;
}

__device__ __forceinline__ float discount_factor(
    const FlatCurveParameters& parameters,
    float maturity_years
) {
    return expf(log_discount_factor(parameters, maturity_years));
}

__device__ __forceinline__ float instantaneous_forward(
    const FlatCurveParameters& parameters,
    float
) {
    return parameters.rate;
}

__device__ __forceinline__ float forward_derivative(
    const FlatCurveParameters&,
    float
) {
    return 0.0f;
}

__device__ __forceinline__ float forward_rate(
    const FlatCurveParameters& parameters,
    float,
    float
) {
    return parameters.rate;
}

}  // namespace ai_factory::workbench::curve::flat
