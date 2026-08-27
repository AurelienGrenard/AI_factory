// Schobel-Zhu parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::schobel_zhu {

struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_volatility;
    float mean_reversion;
    float long_run_volatility;
    float volatility_of_volatility;
    float correlation;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
