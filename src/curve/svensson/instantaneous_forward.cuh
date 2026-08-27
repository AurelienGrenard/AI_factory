// Host/device Svensson instantaneous-forward formula.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench::curve::svensson {
namespace detail {

__host__ __device__ __forceinline__ float forward_exponential(float value) {
    return expf(value);
}

__host__ __device__ __forceinline__ double forward_exponential(double value) {
    return exp(value);
}

}  // namespace detail

// Evaluate f(0,T) directly from one Svensson parameter set.
template<typename Real>
__host__ __device__ __forceinline__ Real instantaneous_forward_formula(
    Real beta0,
    Real beta1,
    Real beta2,
    Real beta3,
    Real tau1,
    Real tau2,
    Real maturity
) {
    const Real x1 = maturity / tau1;
    const Real x2 = maturity / tau2;
    return beta0
        + detail::forward_exponential(-x1) * (beta1 + beta2 * x1)
        + beta3 * x2 * detail::forward_exponential(-x2);
}

}  // namespace ai_factory::workbench::curve::svensson
