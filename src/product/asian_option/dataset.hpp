// Asian-option dataset row and host-side JSON loader.
#pragma once

#include "product/asian_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Asian-option row into one contiguous vector.
std::vector<AsianOptionParameters> load_asian_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
