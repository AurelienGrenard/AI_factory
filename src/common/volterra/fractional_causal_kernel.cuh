// Cell averages for causal simulations of the normalized fractional kernel.
#pragma once

#include "common/volterra/fractional_kernel.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench::volterra {

// K_H(t) = t^(H-1/2) / Gamma(H+1/2).  The Gaussian rough-Bergomi
// normalization lives in FractionalHybridKernelPolicy instead.
struct FractionalCausalKernelPolicy {
    using Parameters = float;

    struct PreparedKernel {
        FractionalPowerKernel power;
        float time_step;
        float time_step_to_exponent;
        float inverse_gamma;
        float first_cell_integral;
        float singular_independent_loading;
    };

    __host__ __device__ static PreparedKernel prepare(float hurst, float dt) {
        const FractionalPowerKernel power =
            FractionalPowerKernel::prepare(hurst);
        const float inverse_gamma = 1.0f / tgammaf(hurst + 0.5f);
        const float first_cell_integral = powf(dt, hurst + 0.5f)
            * inverse_gamma / (hurst + 0.5f);
        const float stochastic_cell_variance = powf(dt, 2.0f * hurst)
            * inverse_gamma * inverse_gamma / (2.0f * hurst);
        const float projected_variance = first_cell_integral
            * first_cell_integral / dt;
        return {
            power, dt, powf(dt, hurst - 0.5f), inverse_gamma,
            first_cell_integral,
            sqrtf(fmaxf(stochastic_cell_variance - projected_variance, 0.0f)),
        };
    }

    __host__ __device__ static float cell_average_weight(
        const PreparedKernel& kernel, unsigned int lag
    ) {
        return kernel.inverse_gamma
            * kernel.power.cell_average_weight_from_scale(
                kernel.time_step_to_exponent, lag
            );
    }
};

}  // namespace ai_factory::workbench::volterra
