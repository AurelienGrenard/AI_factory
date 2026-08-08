// Variance-Gamma model row shared by dataset and CUDA code.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::variance_gamma {

// Risk-neutral exponential Variance-Gamma inputs in the Madan-Carr-Chang form.
struct VarianceGammaModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float sigma;
    float nu;
    float theta;
};

static_assert(std::is_trivially_copyable_v<VarianceGammaModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<VarianceGammaModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::variance_gamma
