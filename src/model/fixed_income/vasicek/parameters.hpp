// Vasicek parameters shared by host loaders, analytics, and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::fixed_income::vasicek {

// Coefficients in dr_t = a * (b - r_t) dt + sigma dW_t.
struct ProcessParameters {
    float mean_reversion;
    float long_term_mean;
    float volatility;
};

// Standalone Vasicek short-rate process together with r(0).
struct ModelParameters {
    ProcessParameters process;
    float initial_state;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::fixed_income::vasicek
