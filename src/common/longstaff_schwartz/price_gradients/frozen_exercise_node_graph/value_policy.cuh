// Scalar node values produced by frozen-exercise replay.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

struct FrozenExerciseValueNodePolicy {
    using NodeValue = float;
    struct Metadata {};

    __device__ __forceinline__ static float payoff(
        const Metadata&,
        float value
    ) {
        return value;
    }

    __device__ __forceinline__ static float centered_first(
        const Metadata&,
        const Metadata&,
        float first,
        float second,
        float represented_width
    ) {
        return (second - first) / represented_width;
    }
};

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
