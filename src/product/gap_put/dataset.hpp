// Gap-put dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Gap-put parameters stored contiguously for CUDA.
struct GapPutParameters {
    float trigger_strike;
    float payoff_strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<GapPutParameters>);

// Load every gap-put row into one contiguous FP32 vector.
std::vector<GapPutParameters> load_gap_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
