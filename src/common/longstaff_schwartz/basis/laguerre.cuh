// Laguerre bases used by small Longstaff-Schwartz regressions.
#pragma once

#include "common/longstaff_schwartz/basis/feature_vector.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz::basis {

__device__ __forceinline__ float laguerre_0(float) {
    return 1.0f;
}

__device__ __forceinline__ float laguerre_1(float value) {
    return 1.0f - value;
}

__device__ __forceinline__ float laguerre_2(float value) {
    return fmaf(0.5f * value, value, 1.0f - 2.0f * value);
}

// Laguerre primary factor with polynomial secondary-state interactions.
struct LaguerrePolynomialTwoFactorBasis {
    using Input = TwoFactorInput;
    using Features = FeatureVector<6U>;

    static constexpr std::size_t kSize = Features::kSize;

    __device__ __forceinline__ static Features evaluate(
        const Input& input
    ) {
        const float primary_1 = laguerre_1(input.primary);
        return {{
            laguerre_0(input.primary),
            primary_1,
            laguerre_2(input.primary),
            input.secondary,
            input.secondary * input.secondary,
            primary_1 * input.secondary,
        }};
    }
};

// Laguerre family L_0, ..., L_Degree for one positive state.
template<std::size_t Degree>
struct OneFactorLaguerreBasis {
    using Input = float;
    using Features = FeatureVector<Degree + 1U>;

    static constexpr std::size_t kSize = Features::kSize;

    __device__ __forceinline__ static Features evaluate(float value) {
        Features features{};
        features.values[0] = 1.0f;
        if constexpr (Degree >= 1U) {
            features.values[1] = 1.0f - value;
        }
        #pragma unroll
        for (std::size_t degree = 2U; degree <= Degree; ++degree) {
            const float n = static_cast<float>(degree);
            features.values[degree] = (
                (2.0f * n - 1.0f - value) * features.values[degree - 1U]
                - (n - 1.0f) * features.values[degree - 2U]
            ) / n;
        }
        return features;
    }
};

}  // namespace ai_factory::workbench::longstaff_schwartz::basis
