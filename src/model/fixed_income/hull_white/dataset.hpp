// Hull-White dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::hull_white {

// Model parameters independent of the initial discount curve.
struct ModelParameters {
    float mean_reversion;
    float volatility;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::hull_white
