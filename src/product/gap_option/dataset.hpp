// Gap-option dataset row and host-side JSON loader.
#pragma once

#include "common/option_side.cuh"

#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Gap-option parameters stored contiguously for CUDA.
struct GapOptionParameters {
    float trigger_strike;
    float payoff_strike;
    std::uint32_t maturity;
};

static_assert(std::is_trivially_copyable_v<GapOptionParameters>);

// Load and validate every gap-option row for the selected payoff side.
std::vector<GapOptionParameters> load_gap_options(
    const std::filesystem::path& dataset_path,
    OptionSide side
);

}  // namespace ai_factory::workbench::product
