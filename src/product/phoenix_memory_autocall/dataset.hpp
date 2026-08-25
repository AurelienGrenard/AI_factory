// Phoenix-Memory-autocall dataset row and host-side JSON loader.
#pragma once

#include "product/phoenix_memory_autocall/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every Phoenix-Memory-autocall row into one contiguous vector.
std::vector<PhoenixMemoryAutocallParameters> load_phoenix_memory_autocalls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
