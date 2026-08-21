// Lookback-option dataset row and host-side JSON loader.
#pragma once

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// LookbackOptionParameters is the compact product row transferred to CUDA.
struct LookbackOptionParameters {
    float strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<LookbackOptionParameters>);

// Load every lookback-option row into one contiguous vector.
std::vector<LookbackOptionParameters> load_lookback_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
