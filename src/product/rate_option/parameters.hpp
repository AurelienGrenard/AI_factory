// Forward-rate-option contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct RateOptionParameters {
    float notional;
    float strike;
    std::uint32_t fixing_time;
    std::uint32_t payment_time;
    std::uint32_t accrual_period;
};

static_assert(std::is_trivially_copyable_v<RateOptionParameters>);

}  // namespace ai_factory::workbench::product
