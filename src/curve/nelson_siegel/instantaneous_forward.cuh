// Host/device Nelson-Siegel instantaneous-forward formula.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench::curve::nelson_siegel {
namespace detail {

__host__ __device__ __forceinline__ float forward_exponential(float value) {
    return expf(value);
}

__host__ __device__ __forceinline__ double forward_exponential(double value) {
    return exp(value);
}

}  // namespace detail

// Evaluate f(0,T) from beta values and the dimensionless maturity T / tau.
template<typename Real>
__host__ __device__ __forceinline__ Real instantaneous_forward_formula(
    Real beta0,
    Real beta1,
    Real beta2,
    Real scaled_maturity
) {
    return beta0
        + detail::forward_exponential(-scaled_maturity)
            * (beta1 + beta2 * scaled_maturity);
}

}  // namespace ai_factory::workbench::curve::nelson_siegel
