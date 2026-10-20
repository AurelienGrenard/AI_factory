#pragma once

#include "model/fixed_income/constant_rate/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::fixed_income::constant_rate {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path);

}  // namespace ai_factory::workbench::model::fixed_income::constant_rate
