// Normal-Inverse-Gaussian model row shared by dataset and CUDA code.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::normal_inverse_gaussian {

// Risk-neutral exponential NIG inputs without a redundant location parameter.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float alpha;
    float beta;
    float delta;
};

static_assert(
    std::is_trivially_copyable_v<ModelParameters>
);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::normal_inverse_gaussian
