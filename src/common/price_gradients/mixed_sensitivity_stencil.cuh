// Tensor-product stencil for one selected mixed second derivative.
#pragma once

#include "common/price_gradients/sensitivity_stencil.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::price_gradients {

inline constexpr std::size_t kMixedSensitivityNodeCapacity = 9U;

struct FirstSensitivitySupport {
    std::uint8_t local_nodes[3U]{};
    float weights[3U]{};
    std::uint8_t node_count = 0U;
};

template<std::size_t NodeCapacity>
__host__ __device__ inline FirstSensitivitySupport first_sensitivity_support(
    const SensitivityStencil<NodeCapacity>& stencil
) {
    FirstSensitivitySupport result{};
    if (stencil.kind == StencilKind::centered) {
        result.local_nodes[0U] = 1U;
        result.local_nodes[1U] = 2U;
        result.weights[0U] = -1.0f / stencil.represented_width;
        result.weights[1U] = 1.0f / stencil.represented_width;
        result.node_count = 2U;
        return result;
    }

    result.local_nodes[0U] = 0U;
    result.local_nodes[1U] = 1U;
    result.local_nodes[2U] = 2U;
    result.weights[0U] = -(
        stencil.first_endpoint_weights[0U]
        + stencil.first_endpoint_weights[1U]
    );
    result.weights[1U] = stencil.first_endpoint_weights[0U];
    result.weights[2U] = stencil.first_endpoint_weights[1U];
    result.node_count = 3U;
    return result;
}

struct MixedSensitivityStencil {
    float weights[kMixedSensitivityNodeCapacity]{};
    std::uint8_t first_local_nodes[kMixedSensitivityNodeCapacity]{};
    std::uint8_t second_local_nodes[kMixedSensitivityNodeCapacity]{};
    std::uint8_t node_count = 0U;
};

template<std::size_t FirstCapacity, std::size_t SecondCapacity>
__host__ __device__ inline MixedSensitivityStencil
make_mixed_sensitivity_stencil(
    const SensitivityStencil<FirstCapacity>& first,
    const SensitivityStencil<SecondCapacity>& second
) {
    const auto first_support = first_sensitivity_support(first);
    const auto second_support = first_sensitivity_support(second);
    MixedSensitivityStencil result{};
    for (std::size_t i = 0U; i < first_support.node_count; ++i) {
        for (std::size_t j = 0U; j < second_support.node_count; ++j) {
            const auto node = result.node_count++;
            result.first_local_nodes[node] = first_support.local_nodes[i];
            result.second_local_nodes[node] = second_support.local_nodes[j];
            result.weights[node] =
                first_support.weights[i] * second_support.weights[j];
        }
    }
    return result;
}

static_assert(std::is_trivially_copyable_v<FirstSensitivitySupport>);
static_assert(std::is_trivially_copyable_v<MixedSensitivityStencil>);

}  // namespace ai_factory::workbench::price_gradients
