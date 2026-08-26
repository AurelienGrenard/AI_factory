// Merton model parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::merton {

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

}  // namespace ai_factory::workbench::model::equity::merton
