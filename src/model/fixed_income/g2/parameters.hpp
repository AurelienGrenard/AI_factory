// G2 parameters shared by host loaders, analytics, and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::g2 {

// Coefficients of two correlated centered Ornstein-Uhlenbeck factors.
struct ProcessParameters {
    float mean_reversion_x;
    float volatility_x;
    float mean_reversion_y;
    float volatility_y;
    float correlation;
};

// Two Gaussian factor states whose sum is the short rate.
struct State {
    float state_x;
    float state_y;
};

// Standalone G2 short-rate model combining the process with both initial states.
struct ModelParameters {
    ProcessParameters process;
    State initial_state;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::g2
