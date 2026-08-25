// Up-and-out-option dataset row and host-side JSON loader.
#pragma once

#include "product/up_and_out_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Up-and-out-option row into one contiguous vector.
std::vector<UpAndOutOptionParameters> load_up_and_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
