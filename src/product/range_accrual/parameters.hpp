// Range-accrual contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct RangeAccrualParameters {
    std::uint32_t maturity;
    std::uint32_t observation_interval;
    float lower_barrier;
    float upper_barrier;
    float coupon_rate;
};

static_assert(std::is_trivially_copyable_v<RangeAccrualParameters>);

}  // namespace ai_factory::workbench::product
