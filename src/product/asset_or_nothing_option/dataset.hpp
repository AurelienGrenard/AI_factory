// Asset-or-nothing dataset row and host-side JSON loader.
#pragma once

#include "product/asset_or_nothing_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every asset-or-nothing row into one contiguous vector.
std::vector<AssetOrNothingOptionParameters> load_asset_or_nothing_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
