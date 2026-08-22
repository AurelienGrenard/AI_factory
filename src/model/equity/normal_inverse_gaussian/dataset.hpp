// Normal-Inverse-Gaussian host-side JSON loader.
#pragma once

#include "model/equity/normal_inverse_gaussian/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::normal_inverse_gaussian {

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::normal_inverse_gaussian
