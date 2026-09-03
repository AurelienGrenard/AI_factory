// G2 parameters shared by host loaders, analytics, and CUDA dynamics.
#pragma once

#include "model/fixed_income/g2/state.hpp"

#include <type_traits>

namespace ai_factory::workbench::model::fixed_income::g2 {

// Coefficients of two correlated centered Ornstein-Uhlenbeck factors.
struct ProcessParameters {
    float mean_reversion_x;
    float volatility_x;
    float mean_reversion_y;
    float volatility_y;
    float correlation;
};

// Standalone G2 short-rate model combining the process with both initial states.
struct ModelParameters {
    ProcessParameters process;
    State initial_state;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::fixed_income::g2
