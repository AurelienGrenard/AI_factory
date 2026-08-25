// Geometric-Asian-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct GeometricAsianOptionParameters {
    float strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<GeometricAsianOptionParameters>);

}  // namespace ai_factory::workbench::product
