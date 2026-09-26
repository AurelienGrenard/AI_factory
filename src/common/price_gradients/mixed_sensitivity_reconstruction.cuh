// Shared pathwise reconstruction of one mixed second derivative.
#pragma once

#include "common/price_gradients/mixed_sensitivity_stencil.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace ai_factory::workbench::price_gradients {

struct MixedSensitivityValues {
    float values[kMixedSensitivityNodeCapacity]{};

    __host__ __device__ float& operator[](std::size_t index) {
        return values[index];
    }

    __host__ __device__ float operator[](std::size_t index) const {
        return values[index];
    }
};

__host__ __device__ inline float reconstruct_mixed_sensitivity(
    const MixedSensitivityStencil& stencil,
    const MixedSensitivityValues& values,
    float central_value
) {
    float result = 0.0f;
    for (std::size_t node = 0U; node < stencil.node_count; ++node) {
        result = fmaf(
            stencil.weights[node],
            values[node] - central_value,
            result
        );
    }
    return result;
}

}  // namespace ai_factory::workbench::price_gradients
