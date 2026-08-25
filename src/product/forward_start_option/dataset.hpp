// Forward-start-option dataset row and host-side JSON loader.
#pragma once

#include "product/forward_start_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Forward-start-option row into one contiguous vector.
std::vector<ForwardStartOptionParameters> load_forward_start_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
