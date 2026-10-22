// Fixed-size sums of exponentials for Markovian Volterra lifts.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace ai_factory::workbench::volterra {

template<std::size_t FactorCount>
requires (FactorCount > 0U)
struct ExponentialKernel {
    float nodes[FactorCount];
    float weights[FactorCount];

    __host__ __device__ float weight_sum() const {
        float result = 0.0f;
        #pragma unroll
        for (std::size_t factor = 0U; factor < FactorCount; ++factor)
            result += weights[factor];
        return result;
    }

    __host__ __device__ float value(float time_years) const {
        float result = 0.0f;
        #pragma unroll
        for (std::size_t factor = 0U; factor < FactorCount; ++factor)
            result += weights[factor] * expf(-nodes[factor] * time_years);
        return result;
    }
};

static_assert(std::is_trivially_copyable_v<ExponentialKernel<2U>>);

}  // namespace ai_factory::workbench::volterra
