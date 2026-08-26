// Black-Scholes host-side JSON loader.
#pragma once

#include "model/equity/black_scholes/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::model::equity::black_scholes {

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
