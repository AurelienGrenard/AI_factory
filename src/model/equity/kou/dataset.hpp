// Kou host-side JSON loader.
#pragma once

#include "model/equity/kou/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::kou {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::kou
