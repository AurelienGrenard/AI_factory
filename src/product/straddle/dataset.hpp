// Straddle dataset row and host-side JSON loader.
#pragma once

#include "product/straddle/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every straddle row into one contiguous vector.
std::vector<StraddleParameters> load_straddles(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
