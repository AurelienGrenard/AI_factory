// Host FP64 and host/device FP32 Nelson-Siegel instantaneous-forward formula.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench::curve::nelson_siegel {
// Evaluate f(0,T) from beta values and the dimensionless maturity T / tau.
__host__ __device__ __forceinline__ float instantaneous_forward_formula(
    float beta0,
    float beta1,
    float beta2,
    float scaled_maturity
) {
    return beta0
        + expf(-scaled_maturity)
            * (beta1 + beta2 * scaled_maturity);
}

// Dataset generation may evaluate extrema in FP64, but this overload is not
// part of the device API.
inline double instantaneous_forward_formula(
    double beta0,
    double beta1,
    double beta2,
    double scaled_maturity
) {
    return beta0
        + std::exp(-scaled_maturity)
            * (beta1 + beta2 * scaled_maturity);
}

}  // namespace ai_factory::workbench::curve::nelson_siegel
