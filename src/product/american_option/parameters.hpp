// American-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct AmericanOptionParameters {
    float strike;
    std::uint32_t maturity_days;
    std::uint32_t exercise_interval_days;
};

static_assert(std::is_trivially_copyable_v<AmericanOptionParameters>);

}  // namespace ai_factory::workbench::product
