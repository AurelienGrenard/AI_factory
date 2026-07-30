// Nelson-Siegel parameters and host-side dataset loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::curve {

// One compact FP32 Nelson-Siegel curve parameterization.
struct NelsonSiegelParameters {
    float beta0;
    float beta1;
    float beta2;
    float tau;
};

static_assert(std::is_trivially_copyable_v<NelsonSiegelParameters>);

// Load every curve row from JSON into one contiguous FP32 vector.
std::vector<NelsonSiegelParameters> load_nelson_siegel(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::curve
