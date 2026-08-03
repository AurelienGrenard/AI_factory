// Asset-or-nothing dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

// Asset-or-nothing call parameters stored contiguously for CUDA.
struct AssetOrNothingCallParameters {
    float strike;
    float maturity;
};

static_assert(std::is_trivially_copyable_v<AssetOrNothingCallParameters>);

// Load every asset-or-nothing row into one contiguous FP32 vector.
std::vector<AssetOrNothingCallParameters> load_asset_or_nothing_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
