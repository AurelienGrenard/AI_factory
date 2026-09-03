#pragma once

#include "model/equity/markovian/stein_stein/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::stein_stein {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::stein_stein
