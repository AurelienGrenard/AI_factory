// Athena-autocall dataset row and host-side JSON loader.
#pragma once

#include "product/athena_autocall/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Athena-autocall row into one contiguous vector.
std::vector<AthenaAutocallParameters> load_athena_autocalls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
