// Phoenix-Memory-autocall dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact memory-coupon issuance terms transferred to the CUDA pricer.
struct PhoenixMemoryAutocallParameters {
    float maturity;
    float observation_interval;
    float autocall_barrier;
    float coupon_barrier;
    float protection_barrier;
    float annual_coupon_rate;
};

static_assert(std::is_trivially_copyable_v<PhoenixMemoryAutocallParameters>);

// Load every Phoenix-Memory-autocall row into one contiguous FP32 vector.
std::vector<PhoenixMemoryAutocallParameters> load_phoenix_memory_autocalls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
