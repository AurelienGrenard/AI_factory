// Ornstein-Uhlenbeck dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Curve-independent coefficients shared with models built from OU dynamics.
struct OrnsteinUhlenbeckDynamicsParameters {
    float mean_reversion;
    float volatility;
};

// Standalone OU short-rate row with its initial factor value.
struct OrnsteinUhlenbeckModelParameters {
    OrnsteinUhlenbeckDynamicsParameters dynamics;
    float initial_factor;
};

static_assert(std::is_trivially_copyable_v<OrnsteinUhlenbeckModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<OrnsteinUhlenbeckModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
