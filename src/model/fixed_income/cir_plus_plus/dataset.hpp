// CIR++ factor host-side JSON loader.
#pragma once

#include "model/fixed_income/cir_plus_plus/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus {

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus
