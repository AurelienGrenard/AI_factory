// Range Accrual dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact observation-band terms transferred to the CUDA pricer.
struct RangeAccrualParameters {
    std::uint32_t maturity;
    std::uint32_t observation_interval;
    float lower_barrier;
    float upper_barrier;
    float coupon_rate;
};

static_assert(std::is_trivially_copyable_v<RangeAccrualParameters>);

// Load every Range Accrual row into one contiguous vector.
std::vector<RangeAccrualParameters> load_range_accruals(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
