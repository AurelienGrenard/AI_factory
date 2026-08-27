// Hermite bases for normalized Gaussian continuation states.
#pragma once

#include "common/longstaff_schwartz/basis/feature_vector.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz::basis {

// Probabilists' Hermite family H_0, ..., H_Degree.
template<std::size_t Degree>
struct OneFactorHermiteBasis {
    using Input = float;
    using Features = FeatureVector<Degree + 1U>;

    static constexpr std::size_t kSize = Features::kSize;

    __device__ __forceinline__ static Features evaluate(float value) {
        Features features{};
        features.values[0] = 1.0f;
        if constexpr (Degree >= 1U) {
            features.values[1] = value;
        }
        #pragma unroll
        for (std::size_t degree = 2U; degree <= Degree; ++degree) {
            features.values[degree] = fmaf(
                value,
                features.values[degree - 1U],
                -static_cast<float>(degree - 1U)
                    * features.values[degree - 2U]
            );
        }
        return features;
    }
};

// Quadratic tensor subset for two standardized Gaussian factors.
struct TwoFactorHermiteBasis {
    using Input = TwoFactorInput;
    using Features = FeatureVector<6U>;

    static constexpr std::size_t kSize = Features::kSize;

    __device__ __forceinline__ static Features evaluate(
        const Input& input
    ) {
        return {{
            1.0f,
            input.primary,
            input.secondary,
            fmaf(input.primary, input.primary, -1.0f),
            input.primary * input.secondary,
            fmaf(input.secondary, input.secondary, -1.0f),
        }};
    }
};

}  // namespace ai_factory::workbench::longstaff_schwartz::basis
