// Merton jump-diffusion model row shared by dataset and CUDA code.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::merton {

struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float volatility;
    float jump_intensity;
    float jump_log_mean;
    float jump_log_volatility;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::merton
