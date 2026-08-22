// Bates host-side JSON loader.
#pragma once

#include "model/equity/bates/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::bates {

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::bates
