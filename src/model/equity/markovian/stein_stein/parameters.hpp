// Compact host/device parameter row for the Stein--Stein model.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::stein_stein {

struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_volatility;
    float mean_reversion;
    float volatility_of_volatility;
    float rho;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::stein_stein
