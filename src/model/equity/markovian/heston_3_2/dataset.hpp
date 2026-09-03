#pragma once

#include "model/equity/markovian/heston_3_2/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::heston_3_2 {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::heston_3_2
