// Piecewise-linear hinge bases with compile-time knot sets.
#pragma once

#include "common/longstaff_schwartz/basis/feature_vector.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <concepts>
#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz::basis {

struct StandardizedFiveKnotSet {
    static constexpr std::size_t kSize = 5U;

    __device__ __forceinline__ static float center(std::size_t index) {
        switch (index) {
            case 0U: return -2.0f;
            case 1U: return -1.0f;
            case 2U: return 0.0f;
            case 3U: return 1.0f;
            default: return 2.0f;
        }
    }
};

template<typename KnotSet>
concept HingeKnotSet = requires(std::size_t index) {
    { KnotSet::kSize } -> std::convertible_to<std::size_t>;
    { KnotSet::center(index) } -> std::same_as<float>;
};

template<HingeKnotSet KnotSet>
struct CenteredHingeBasis {
    using Input = float;
    using Features = FeatureVector<KnotSet::kSize + 2U>;

    static constexpr std::size_t kSize = Features::kSize;

    __device__ __forceinline__ static Features evaluate(float value) {
        Features features{};
        features.values[0] = 1.0f;
        features.values[1] = value;
        #pragma unroll
        for (std::size_t knot = 0U; knot < KnotSet::kSize; ++knot) {
            features.values[knot + 2U] = fmaxf(
                value - KnotSet::center(knot), 0.0f
            );
        }
        return features;
    }
};

using StandardizedFiveKnotHingeBasis =
    CenteredHingeBasis<StandardizedFiveKnotSet>;

}  // namespace ai_factory::workbench::longstaff_schwartz::basis
