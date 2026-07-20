// Public Heston data representation shared by every Heston product pricer.
// It also declares the host loader that converts a registry JSON directly into
// the contiguous FP32 array copied to the GPU.
#pragma once

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::heston {

// HestonModelInput is the compact per-row model layout transferred to CUDA.
struct HestonModelInput {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_variance;
    float kappa;
    float theta;
    float gamma;
    float rho;
};

// Load and convert every model row from a Heston registry JSON database.
std::vector<HestonModelInput> load_heston(
    const std::filesystem::path& json_path
);

}  // namespace ai_factory::workbench::heston
