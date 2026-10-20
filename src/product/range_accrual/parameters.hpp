// Range-accrual contract parameters transferred to CUDA.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

struct RangeAccrualParameters {
    // Barriers are intrinsic ratios S(t)/S(0), not quoted spot levels.
    static constexpr bool kAlreadyRelativePriceLevels = true;
    std::uint32_t maturity_days;
    std::uint32_t observation_interval_days;
    float lower_barrier;
    float upper_barrier;
    float coupon_rate;
};

static_assert(std::is_trivially_copyable_v<RangeAccrualParameters>);

}  // namespace ai_factory::workbench::product
