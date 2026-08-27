// Fractional power-kernel coefficients shared by Volterra schemes.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench::volterra {

// K_H(t) = t^(H-1/2). Normalizations such as sqrt(2H) remain a model choice.
struct FractionalPowerKernel {
    float hurst_exponent;
    float exponent;
    float exponent_plus_one;
    float two_h;

    __host__ __device__ static FractionalPowerKernel prepare(float h) {
        return {h, h - 0.5f, h + 0.5f, 2.0f * h};
    }

    // Average K over the cell whose one-based lag is `lag`.
    __host__ __device__ float cell_average_weight(
        float dt,
        unsigned int lag
    ) const {
        return cell_average_weight_from_scale(powf(dt, exponent), lag);
    }

    // Reuse a precomputed dt^exponent when a row already stores it. Besides
    // avoiding a transcendental, this preserves the caller's FP32 operation
    // order across direct and FFT implementations.
    __host__ __device__ float cell_average_weight_from_scale(
        float dt_to_exponent,
        unsigned int lag
    ) const {
        const float upper = static_cast<float>(lag);
        const float lower = static_cast<float>(lag - 1U);
        return dt_to_exponent
            * (powf(upper, exponent_plus_one)
               - powf(lower, exponent_plus_one))
            / exponent_plus_one;
    }

    __host__ __device__ float normalized_variance(float time) const {
        return powf(time, two_h);
    }
};

}  // namespace ai_factory::workbench::volterra
