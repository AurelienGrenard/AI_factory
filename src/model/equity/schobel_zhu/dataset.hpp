// Schobel-Zhu stochastic-volatility model row.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::schobel_zhu {

struct SchobelZhuModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_volatility;
    float mean_reversion;
    float long_run_volatility;
    float volatility_of_volatility;
    float correlation;
};

static_assert(std::is_trivially_copyable_v<SchobelZhuModelParameters>);

std::vector<SchobelZhuModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::schobel_zhu
