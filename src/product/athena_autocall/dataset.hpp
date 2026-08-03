// Athena-autocall dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Compact accumulated-gain issuance terms transferred to the CUDA pricer.
struct AthenaAutocallParameters {
    float maturity;
    float observation_interval;
    float autocall_barrier;
    float protection_barrier;
    float annual_coupon_rate;
};

static_assert(std::is_trivially_copyable_v<AthenaAutocallParameters>);

// Load every Athena-autocall row into one contiguous FP32 vector.
std::vector<AthenaAutocallParameters> load_athena_autocalls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
