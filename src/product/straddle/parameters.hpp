// Straddle contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct StraddleParameters {
    float strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<StraddleParameters>);

}  // namespace ai_factory::workbench::product
