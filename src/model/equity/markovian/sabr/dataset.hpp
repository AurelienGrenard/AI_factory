// Loader declarations for SABR parameter datasets.
#pragma once

#include "model/equity/markovian/sabr/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::sabr {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::sabr
