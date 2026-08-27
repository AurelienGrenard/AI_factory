// Rough-Heston host-side JSON loader.
#pragma once

#include "model/equity/rough/rough_heston/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::rough_heston {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::rough_heston
