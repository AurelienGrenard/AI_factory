// Host-side JSON loader for flat curve rows.
#pragma once

#include "curve/flat/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::curve::flat {

// Load every curve row from JSON into one contiguous FP32 vector.
std::vector<FlatCurveParameters> load_curves(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::curve::flat
