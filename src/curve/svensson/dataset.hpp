// Host-side JSON loader for Svensson curve rows.
#pragma once

#include "curve/svensson/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::curve::svensson {

// Load every curve row from JSON into one contiguous FP32 vector.
std::vector<SvenssonParameters> load_curves(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::curve::svensson
