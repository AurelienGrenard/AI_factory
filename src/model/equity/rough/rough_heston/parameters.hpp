// Rough-Heston parameters shared by host preparation and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::rough_heston {

// Volterra variance drift is theta - lambda * V. `theta` is therefore the
// constant variance drift, not the long-run variance theta / lambda.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_variance;
    float mean_reversion;
    float variance_drift;
    float volatility_of_variance;
    float hurst_exponent;
    float rho;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::rough_heston
