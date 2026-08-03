// Asset-or-nothing dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Asset-or-nothing put parameters stored contiguously for CUDA.
struct AssetOrNothingPutParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<AssetOrNothingPutParameters>);

// Load every asset-or-nothing row into one contiguous FP32 vector.
std::vector<AssetOrNothingPutParameters> load_asset_or_nothing_puts(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
