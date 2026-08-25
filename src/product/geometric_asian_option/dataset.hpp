// Geometric-Asian-option dataset row and host-side JSON loader.
#pragma once

#include "product/geometric_asian_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Geometric-Asian-option row into one contiguous vector.
std::vector<GeometricAsianOptionParameters> load_geometric_asian_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
