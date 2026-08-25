// Asset-or-nothing-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct AssetOrNothingOptionParameters {
    float strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<AssetOrNothingOptionParameters>);

}  // namespace ai_factory::workbench::product
