// Device primitives for the standard normal distribution.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench {

// Evaluate the standard-normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    constexpr float inverse_sqrt_two = 0.70710678118654752440f;
    return 0.5f * erfcf(-value * inverse_sqrt_two);
}

}  // namespace ai_factory::workbench
