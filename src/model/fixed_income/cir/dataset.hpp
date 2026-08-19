// CIR dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::cir {

// Coefficients in dr_t = kappa * (theta - r_t) dt + sigma sqrt(r_t) dW_t.
struct CirProcessParameters {
    float mean_reversion;
    float long_term_mean;
    float volatility;
};

// Standalone CIR short-rate process together with r(0).
struct CirModelParameters {
    CirProcessParameters process;
    float initial_state;
};

static_assert(std::is_trivially_copyable_v<CirModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<CirModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::cir
