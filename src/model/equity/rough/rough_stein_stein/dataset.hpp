// Loader declarations for rough Stein--Stein parameter datasets.
#pragma once

#include "model/equity/rough/rough_stein_stein/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::rough_stein_stein {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
