// American-option dataset row and host-side JSON loader.
#pragma once

#include "product/american_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every American-option row into one contiguous vector.
std::vector<AmericanOptionParameters> load_american_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
