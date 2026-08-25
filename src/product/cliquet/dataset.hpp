// Cliquet dataset row and host-side JSON loader.
#pragma once

#include "product/cliquet/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Cliquet row into one contiguous vector.
std::vector<CliquetParameters> load_cliquets(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
