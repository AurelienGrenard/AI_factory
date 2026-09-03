// Compact host/device parameter row for the SABR model.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::sabr {

// initial_volatility is the time-zero log-return volatility at spot, not the
// dimensional SABR alpha.  This keeps its meaning invariant when spot varies.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_volatility;
    float volatility_of_volatility;
    float rho;
    float beta;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::sabr
