// Public Heston parameters and host-side registry loader.
#pragma once

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::heston {

// Compact FP32 model row transferred from host memory to CUDA.
struct HestonModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_variance;
    float kappa;
    float theta;
    float gamma;
    float rho;
};

// Load every row from a Heston registry JSON into one contiguous vector.
std::vector<HestonModelParameters> load_heston(
    const std::filesystem::path& json_path
);

}  // namespace ai_factory::workbench::heston
