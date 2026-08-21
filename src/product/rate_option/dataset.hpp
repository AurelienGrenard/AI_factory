// Forward-rate-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact caplet/floorlet contract transferred to CUDA.
struct RateOptionParameters {
    float notional;
    float strike;
    std::uint32_t fixing_time;
    std::uint32_t payment_time;
    std::uint32_t accrual_period;
};

static_assert(std::is_trivially_copyable_v<RateOptionParameters>);

// Load every forward-rate-option row into one contiguous vector.
std::vector<RateOptionParameters> load_rate_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
