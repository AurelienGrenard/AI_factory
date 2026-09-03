#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
