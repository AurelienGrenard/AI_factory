// Svensson dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::curve::svensson {

// One compact FP32 Svensson curve parameterization.
struct SvenssonParameters {
    float beta0;
    float beta1;
    float beta2;
    float beta3;
    float tau1;
    float tau2;
};

static_assert(std::is_trivially_copyable_v<SvenssonParameters>);

// Load every curve row from JSON into one contiguous FP32 vector.
std::vector<SvenssonParameters> load_curves(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::curve::svensson
