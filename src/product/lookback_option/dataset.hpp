// Lookback-option dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// LookbackOptionParameters is the compact FP32 product row transferred to CUDA.
struct LookbackOptionParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<LookbackOptionParameters>);

// Load every lookback-option row into one contiguous FP32 vector.
std::vector<LookbackOptionParameters> load_lookback_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
