// Reusable CUDA interface for the analytical Nelson-Siegel curve.
#pragma once

#include "curve/nelson_siegel/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::curve::nelson_siegel {

// Return the continuously compounded zero rate z(0, maturity_years).
__device__ __forceinline__ float zero_rate(
    const NelsonSiegelParameters& parameters,
    float maturity_years
);

// Return log P(0, maturity_years) without exponentiating the discount factor.
__device__ __forceinline__ float log_discount_factor(
    const NelsonSiegelParameters& parameters,
    float maturity_years
);

// Return the discount factor P(0, maturity_years).
__device__ __forceinline__ float discount_factor(
    const NelsonSiegelParameters& parameters,
    float maturity_years
);

// Return the instantaneous forward rate f(0, maturity_years).
__device__ __forceinline__ float instantaneous_forward(
    const NelsonSiegelParameters& parameters,
    float maturity_years
);

// Return the maturity_years derivative of the instantaneous forward rate.
__device__ __forceinline__ float forward_derivative(
    const NelsonSiegelParameters& parameters,
    float maturity_years
);

// Return the continuously compounded forward rate over [start_years, end_years].
__device__ __forceinline__ float forward_rate(
    const NelsonSiegelParameters& parameters,
    float start_years,
    float end_years
);

// Static adapter consumed by fitted short-rate analytics.
struct AnalyticsProvider {
    using Parameters = NelsonSiegelParameters;

    __device__ __forceinline__ static float log_discount_factor(
        const Parameters& parameters,
        float time_years
    ) {
        return nelson_siegel::log_discount_factor(parameters, time_years);
    }

    __device__ __forceinline__ static float instantaneous_forward(
        const Parameters& parameters,
        float time_years
    ) {
        return nelson_siegel::instantaneous_forward(parameters, time_years);
    }
};

}  // namespace ai_factory::workbench::curve::nelson_siegel
