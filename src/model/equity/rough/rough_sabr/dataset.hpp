// Rough-SABR host-side JSON loader.
#pragma once

#include "model/equity/rough/rough_sabr/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::rough_sabr {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::rough_sabr
