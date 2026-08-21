// Phoenix-autocall dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact issuance terms transferred to the future CUDA pricer.
struct PhoenixAutocallParameters {
    std::uint32_t maturity;
    std::uint32_t observation_interval;
    float autocall_barrier;
    float coupon_barrier;
    float protection_barrier;
    float annual_coupon_rate;
};

static_assert(std::is_trivially_copyable_v<PhoenixAutocallParameters>);

// Load every Phoenix-autocall row into one contiguous vector.
std::vector<PhoenixAutocallParameters> load_phoenix_autocalls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
