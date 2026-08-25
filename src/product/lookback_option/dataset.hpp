// Lookback-option dataset row and host-side JSON loader.
#pragma once

#include "product/lookback_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every lookback-option row into one contiguous vector.
std::vector<LookbackOptionParameters> load_lookback_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
