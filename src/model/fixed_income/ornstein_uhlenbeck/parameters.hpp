// OU parameters shared by host loaders, analytics, and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {

// Coefficients in dX_t = -mean_reversion * X_t dt + volatility * dW_t.
struct ProcessParameters {
    float mean_reversion;
    float volatility;
};

// Standalone short-rate model combining the OU process with X_0.
struct ModelParameters {
    ProcessParameters process;
    float initial_state;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
