// Compact host/device parameter row for the Heston 3/2 model.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::heston_3_2 {

struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_variance;
    float mean_reversion;
    float long_run_variance;
    float volatility_of_variance;
    float rho;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::heston_3_2
