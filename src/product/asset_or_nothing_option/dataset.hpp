// Asset-or-nothing dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Asset-or-nothing option parameters stored contiguously for CUDA.
struct AssetOrNothingOptionParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<AssetOrNothingOptionParameters>);

// Load every asset-or-nothing row into one contiguous FP32 vector.
std::vector<AssetOrNothingOptionParameters> load_asset_or_nothing_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
