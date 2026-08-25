// Gap-option dataset row and host-side JSON loader.
#pragma once

#include "product/gap_option/parameters.hpp"

#include "common/option_side.cuh"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load and validate every gap-option row for the selected payoff side.
std::vector<GapOptionParameters> load_gap_options(
    const std::filesystem::path& dataset_path,
    OptionSide side
);

}  // namespace ai_factory::workbench::product
