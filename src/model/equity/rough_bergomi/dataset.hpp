// Rough-Bergomi model row shared by dataset and CUDA code.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::rough_bergomi {

// Flat-forward-variance rough-Bergomi inputs under the risk-neutral measure.
struct RoughBergomiModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float xi_0;
    float eta;
    float hurst_exponent;
    float rho;
};

static_assert(std::is_trivially_copyable_v<RoughBergomiModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<RoughBergomiModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::rough_bergomi
