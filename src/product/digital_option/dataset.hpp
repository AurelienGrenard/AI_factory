// Digital-option dataset row and host-side JSON loader.
#pragma once

#include "product/digital_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every digital-option row into one contiguous vector.
std::vector<DigitalOptionParameters> load_digital_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
