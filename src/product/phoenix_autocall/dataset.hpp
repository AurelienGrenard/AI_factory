// Phoenix-autocall dataset row and host-side JSON loader.
#pragma once

#include "product/phoenix_autocall/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Phoenix-autocall row into one contiguous vector.
std::vector<PhoenixAutocallParameters> load_phoenix_autocalls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
