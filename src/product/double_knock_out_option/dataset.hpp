// Double-knock-out-option dataset row and host-side JSON loader.
#pragma once

#include "product/double_knock_out_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Double-knock-out-option row into one contiguous vector.
std::vector<DoubleKnockOutOptionParameters> load_double_knock_out_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
