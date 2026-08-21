// Heston dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::heston {

// Compact FP32 model row transferred from host memory to CUDA.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_variance;
    float kappa;
    float theta;
    float gamma;
    float rho;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::heston
