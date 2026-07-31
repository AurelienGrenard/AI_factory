// Hull-White dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::hull_white {

// Model parameters independent of the initial discount curve.
struct HullWhiteOneFactorParameters {
    float mean_reversion;
    float volatility;
};

static_assert(std::is_trivially_copyable_v<HullWhiteOneFactorParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<HullWhiteOneFactorParameters> load_hull_white_one_factor(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::hull_white
