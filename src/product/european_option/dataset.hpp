// European-option dataset row and host-side JSON loader.
#pragma once

#include "product/european_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every European-option row into one contiguous vector.
std::vector<EuropeanOptionParameters> load_european_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
