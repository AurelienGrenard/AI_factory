// Athena-autocall dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact accumulated-gain issuance terms transferred to the CUDA pricer.
struct AthenaAutocallParameters {
    std::uint32_t maturity;
    std::uint32_t observation_interval;
    float autocall_barrier;
    float protection_barrier;
    float annual_coupon_rate;
};

static_assert(std::is_trivially_copyable_v<AthenaAutocallParameters>);

// Load every Athena-autocall row into one contiguous vector.
std::vector<AthenaAutocallParameters> load_athena_autocalls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
