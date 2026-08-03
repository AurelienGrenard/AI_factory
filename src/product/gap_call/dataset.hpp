// Gap-call dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Gap-call parameters stored contiguously for CUDA.
struct GapCallParameters {
    float trigger_strike;
    float payoff_strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<GapCallParameters>);

// Load every gap-call row into one contiguous FP32 vector.
std::vector<GapCallParameters> load_gap_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
