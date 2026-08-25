// Forward-rate-option dataset row and host-side JSON loader.
#pragma once

#include "product/rate_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every forward-rate-option row into one contiguous vector.
std::vector<RateOptionParameters> load_rate_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
