// Gap-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct GapOptionParameters {
    float trigger_strike;
    float payoff_strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<GapOptionParameters>);

}  // namespace ai_factory::workbench::product
