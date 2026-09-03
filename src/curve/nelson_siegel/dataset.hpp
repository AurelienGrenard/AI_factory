// Host-side JSON loader for Nelson-Siegel curve rows.
#pragma once

#include "curve/nelson_siegel/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::curve::nelson_siegel {

// Load every curve row from JSON into one contiguous FP32 vector.
std::vector<NelsonSiegelParameters> load_curves(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::curve::nelson_siegel
