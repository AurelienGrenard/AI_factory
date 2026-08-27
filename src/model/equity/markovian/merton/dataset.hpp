// Merton host-side JSON loader.
#pragma once

#include "model/equity/markovian/merton/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::merton {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::merton
