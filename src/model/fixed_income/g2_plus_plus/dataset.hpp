// G2++ dataset row and host-side JSON loader.
#pragma once

#include "model/fixed_income/g2/dataset.hpp"

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::g2_plus_plus {

// Curve-independent process parameters of one centered G2++ model.
struct ModelParameters {
    model::g2::ProcessParameters process;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::g2_plus_plus
