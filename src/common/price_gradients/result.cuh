// Device differences and fixed-cardinality price/gradient values for moment reduction.
#pragma once

#include "common/price_gradients/stencil.hpp"
#include <cuda_runtime.h>

namespace ai_factory::workbench::price_gradients {

__host__ __device__ inline float difference(const Stencil& stencil, float central, float first, float second) {
    // Retain the historical subtraction/division for centered spot parity.
    if (stencil.kind == StencilKind::centered)
        return (second - first) / stencil.represented_width;
    // Difference form annihilates a constant payoff even after coefficient rounding.
    return stencil.first_weight * (first - central) + stencil.second_weight * (second - central);
}

template<unsigned int K>
struct Result {
    float price;
    float gradients[K == 0U ? 1U : K];
};

}  // namespace ai_factory::workbench::price_gradients
