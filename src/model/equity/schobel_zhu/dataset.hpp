// Schobel-Zhu host-side JSON loader.
#pragma once

#include "model/equity/schobel_zhu/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::schobel_zhu {

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::schobel_zhu
