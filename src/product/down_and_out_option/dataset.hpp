// Down-and-out-option dataset row and host-side JSON loader.
#pragma once

#include "product/down_and_out_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Down-and-out-option row into one contiguous vector.
std::vector<DownAndOutOptionParameters> load_down_and_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
