// Represented finite-difference nodes and coefficients for one sensitivity.
#pragma once

#include "common/price_gradients/configuration.hpp"
#include "common/price_gradients/stencil.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::price_gradients {

template<std::size_t NodeCapacity>
struct DiagonalWeightsStorage {};

template<>
struct DiagonalWeightsStorage<4U> {
    float second_weights[4U]{};
    std::uint8_t node_count = 0U;
};

template<std::size_t NodeCapacity>
struct SensitivityStencil : DiagonalWeightsStorage<NodeCapacity> {
    static_assert(NodeCapacity == 3U || NodeCapacity == 4U);

    StencilKind kind = StencilKind::centered;
    float parameter_values[NodeCapacity]{};
    float displacement = 0.0f;
    float represented_width = 0.0f;
    float first_endpoint_weights[2U]{};
};

template<std::size_t NodeCapacity>
__host__ __device__ inline std::size_t active_node_count(
    const SensitivityStencil<NodeCapacity>& stencil
) {
    if constexpr (NodeCapacity == 3U) {
        return 3U;
    } else {
        return stencil.node_count;
    }
}

template<std::size_t NodeCapacity>
__host__ __device__ inline Stencil legacy_stencil(
    const SensitivityStencil<NodeCapacity>& source
) {
    const bool centered = source.kind == StencilKind::centered;
    return {
        source.parameter_values[0U],
        source.parameter_values[1U],
        source.parameter_values[2U],
        source.displacement,
        source.represented_width,
        centered ? 0.0f : source.first_endpoint_weights[0U],
        centered ? 0.0f : source.first_endpoint_weights[1U],
        source.kind,
    };
}

static_assert(std::is_trivially_copyable_v<SensitivityStencil<3U>>);
static_assert(std::is_trivially_copyable_v<SensitivityStencil<4U>>);
static_assert(sizeof(SensitivityStencil<3U>) == sizeof(Stencil));

}  // namespace ai_factory::workbench::price_gradients
