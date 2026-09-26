// Canonical pair of selected sensitivity coordinates.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::price_gradients {

struct SensitivityPair {
    std::uint16_t first = 0U;
    std::uint16_t second = 0U;

    __host__ __device__ friend constexpr bool operator==(
        SensitivityPair,
        SensitivityPair
    ) = default;
};

__host__ __device__ constexpr SensitivityPair canonical_sensitivity_pair(
    std::uint16_t first,
    std::uint16_t second
) {
    return first < second
        ? SensitivityPair{first, second}
        : SensitivityPair{second, first};
}

static_assert(std::is_trivially_copyable_v<SensitivityPair>);

}  // namespace ai_factory::workbench::price_gradients
