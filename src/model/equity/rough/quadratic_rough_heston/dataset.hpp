// Loader declarations for quadratic rough Heston parameter datasets.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
