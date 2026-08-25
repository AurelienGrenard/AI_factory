// Down-and-in-option dataset row and host-side JSON loader.
#pragma once

#include "product/down_and_in_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Down-and-in-option row into one contiguous vector.
std::vector<DownAndInOptionParameters> load_down_and_in_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
