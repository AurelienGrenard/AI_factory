// Compact reconstruction maps from represented stencil nodes to graph nodes.
#pragma once

#include "common/price_gradients/mixed_sensitivity_stencil.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<std::size_t NodeCapacity>
struct SensitivityNodeIndices {
    std::uint16_t values[NodeCapacity]{};

    __host__ __device__ std::uint16_t& operator[](std::size_t index) {
        return values[index];
    }

    __host__ __device__ std::uint16_t operator[](std::size_t index) const {
        return values[index];
    }
};

struct MixedSensitivityNodeIndices {
    std::uint16_t values[pg::kMixedSensitivityNodeCapacity]{};

    __host__ __device__ std::uint16_t& operator[](std::size_t index) {
        return values[index];
    }

    __host__ __device__ std::uint16_t operator[](std::size_t index) const {
        return values[index];
    }
};

static_assert(std::is_trivially_copyable_v<SensitivityNodeIndices<4U>>);
static_assert(std::is_trivially_copyable_v<MixedSensitivityNodeIndices>);

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
