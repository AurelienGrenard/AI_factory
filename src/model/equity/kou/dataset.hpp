// Kou double-exponential jump-diffusion model row.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::kou {

struct KouModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float volatility;
    float jump_intensity;
    float up_probability;
    float positive_jump_rate;
    float negative_jump_rate;
};

static_assert(std::is_trivially_copyable_v<KouModelParameters>);

std::vector<KouModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::kou
