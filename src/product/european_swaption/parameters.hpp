// European-swaption contract parameters transferred to CUDA.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct RegularEuropeanSwaptionParameters {
    float notional;
    float strike;
    float accrual_fraction;
    std::uint32_t exercise_time;
    std::uint32_t payment_interval;
    std::uint32_t payment_count;
};

struct ExplicitEuropeanSwaptionParameters {
    float notional;
    float strike;
    std::uint32_t exercise_time;
    std::uint32_t payment_count;
    std::size_t schedule_offset;
};

static_assert(
    std::is_trivially_copyable_v<RegularEuropeanSwaptionParameters>
);
static_assert(
    std::is_trivially_copyable_v<ExplicitEuropeanSwaptionParameters>
);
static_assert(sizeof(RegularEuropeanSwaptionParameters) == 24U);
static_assert(sizeof(ExplicitEuropeanSwaptionParameters) == 24U);

}  // namespace ai_factory::workbench::product
