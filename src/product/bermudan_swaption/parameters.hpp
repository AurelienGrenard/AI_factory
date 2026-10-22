// Co-terminal Bermudan-swaption contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct BermudanSwaptionParameters {
    float notional;
    float strike;
    float accrual_fraction;
    std::uint32_t first_exercise_time_days;
    std::uint32_t payment_interval_days;
    std::uint32_t payment_count;
    std::uint32_t exercise_count;
    bool quote_is_relative = false;
};

static_assert(std::is_trivially_copyable_v<BermudanSwaptionParameters>);
static_assert(sizeof(BermudanSwaptionParameters) == 32U);

}  // namespace ai_factory::workbench::product
