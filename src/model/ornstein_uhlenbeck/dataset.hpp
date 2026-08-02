// Ornstein-Uhlenbeck dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Coefficients in dX_t = -mean_reversion * X_t dt + volatility * dW_t.
struct OrnsteinUhlenbeckProcessParameters {
    float mean_reversion;
    float volatility;
};

// Standalone short-rate model combining the OU process with X_0.
struct OrnsteinUhlenbeckModelParameters {
    OrnsteinUhlenbeckProcessParameters process;
    float initial_state;
};

static_assert(std::is_trivially_copyable_v<OrnsteinUhlenbeckModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<OrnsteinUhlenbeckModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
