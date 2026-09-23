// Shared finite-difference reconstruction from represented pathwise values.
#pragma once

#include "common/price_gradients/sensitivity_request.hpp"
#include "common/price_gradients/sensitivity_task.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace ai_factory::workbench::price_gradients {

template<std::size_t NodeCapacity>
__host__ __device__ inline float reconstruct_first_sensitivity(
    const SensitivityStencil<NodeCapacity>& stencil,
    const SensitivityValues<NodeCapacity>& values
) {
    if (stencil.kind == StencilKind::centered) {
        return (values[2U] - values[1U]) / stencil.represented_width;
    }
    return fmaf(
        stencil.first_endpoint_weights[0U],
        values[1U] - values[0U],
        stencil.first_endpoint_weights[1U]
            * (values[2U] - values[0U])
    );
}

template<std::size_t NodeCapacity>
__host__ __device__ inline float reconstruct_second_sensitivity(
    const SensitivityStencil<NodeCapacity>& stencil,
    const SensitivityValues<NodeCapacity>& values
) {
    static_assert(NodeCapacity == 4U);
    float result = 0.0f;
    for (std::size_t node = 1U;
         node < active_node_count(stencil);
         ++node) {
        result = fmaf(
            stencil.second_weights[node],
            values[node] - values[0U],
            result
        );
    }
    return result;
}

template<SensitivityOrders Orders, std::size_t NodeCapacity>
__host__ __device__ inline SensitivityResult reconstruct_sensitivity(
    const SensitivityStencil<NodeCapacity>& stencil,
    const SensitivityValues<NodeCapacity>& values
) {
    static_assert(Orders != SensitivityOrders::none);
    SensitivityResult result{};
    if constexpr (requests_first_v<Orders>) {
        result.first = reconstruct_first_sensitivity(stencil, values);
    }
    if constexpr (requests_second_v<Orders>) {
        result.second = reconstruct_second_sensitivity(stencil, values);
    }
    return result;
}

}  // namespace ai_factory::workbench::price_gradients
