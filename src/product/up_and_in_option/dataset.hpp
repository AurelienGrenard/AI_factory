// Up-and-in-option dataset row and host-side JSON loader.
#pragma once

#include "product/up_and_in_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Up-and-in-option row into one contiguous vector.
std::vector<UpAndInOptionParameters> load_up_and_in_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
