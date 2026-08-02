// G2 dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::g2 {

// Coefficients of two correlated centered Ornstein-Uhlenbeck factors.
struct G2ProcessParameters {
    float mean_reversion_x;
    float volatility_x;
    float mean_reversion_y;
    float volatility_y;
    float correlation;
};

// Two Gaussian factor states whose sum is the short rate.
struct G2State {
    float state_x;
    float state_y;
};

// Standalone G2 short-rate model combining the process with both initial states.
struct G2ModelParameters {
    G2ProcessParameters process;
    G2State initial_state;
};

static_assert(std::is_trivially_copyable_v<G2ModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<G2ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::g2
