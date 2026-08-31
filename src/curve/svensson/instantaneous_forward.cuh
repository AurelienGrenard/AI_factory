// Host FP64 and host/device FP32 Svensson instantaneous-forward formula.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench::curve::svensson {
// Evaluate f(0,T) directly from one Svensson parameter set.
__host__ __device__ __forceinline__ float instantaneous_forward_formula(
    float beta0,
    float beta1,
    float beta2,
    float beta3,
    float tau1,
    float tau2,
    float maturity_years
) {
    const float x1 = maturity_years / tau1;
    const float x2 = maturity_years / tau2;
    return beta0
        + expf(-x1) * (beta1 + beta2 * x1)
        + beta3 * x2 * expf(-x2);
}

// Dataset generation may scan the curve in FP64, but this overload is not
// part of the device API.
inline double instantaneous_forward_formula(
    double beta0,
    double beta1,
    double beta2,
    double beta3,
    double tau1,
    double tau2,
    double maturity_years
) {
    const double x1 = maturity_years / tau1;
    const double x2 = maturity_years / tau2;
    return beta0
        + std::exp(-x1) * (beta1 + beta2 * x1)
        + beta3 * x2 * std::exp(-x2);
}

}  // namespace ai_factory::workbench::curve::svensson
