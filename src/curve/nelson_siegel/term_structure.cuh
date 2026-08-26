// Reusable CUDA interface for the analytical Nelson-Siegel curve.
#pragma once

#include "curve/nelson_siegel/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::curve::nelson_siegel {

// Return the continuously compounded zero rate z(0, maturity).
__device__ __forceinline__ float zero_rate(
    const NelsonSiegelParameters& parameters,
    float maturity
);

// Return log P(0, maturity) without exponentiating the discount factor.
__device__ __forceinline__ float log_discount_factor(
    const NelsonSiegelParameters& parameters,
    float maturity
);

// Return the discount factor P(0, maturity).
__device__ __forceinline__ float discount_factor(
    const NelsonSiegelParameters& parameters,
    float maturity
);

// Return the instantaneous forward rate f(0, maturity).
__device__ __forceinline__ float instantaneous_forward(
    const NelsonSiegelParameters& parameters,
    float maturity
);

// Return the maturity derivative of the instantaneous forward rate.
__device__ __forceinline__ float forward_derivative(
    const NelsonSiegelParameters& parameters,
    float maturity
);

// Return the continuously compounded forward rate over [start, end].
__device__ __forceinline__ float forward_rate(
    const NelsonSiegelParameters& parameters,
    float start,
    float end
);

// Static adapter consumed by fitted short-rate analytics.
struct AnalyticsProvider {
    using Parameters = NelsonSiegelParameters;

    __device__ __forceinline__ static float log_discount_factor(
        const Parameters& parameters,
        float time
    ) {
        return nelson_siegel::log_discount_factor(parameters, time);
    }

    __device__ __forceinline__ static float instantaneous_forward(
        const Parameters& parameters,
        float time
    ) {
        return nelson_siegel::instantaneous_forward(parameters, time);
    }
};

}  // namespace ai_factory::workbench::curve::nelson_siegel
