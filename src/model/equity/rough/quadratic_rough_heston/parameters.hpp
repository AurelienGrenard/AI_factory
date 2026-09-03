// Compact host/device parameter row for the quadratic rough Heston model.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

// Restricted quadratic rough-Heston specification of
// Gatheral-Jusselin-Rosenbaum: V=a(Z-b)^2+c and the same Brownian motion
// drives the feedback factor Z and the stock.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float initial_feedback;
    float quadratic_scale;
    float quadratic_shift;
    float variance_floor;
    float feedback_rate;
    float feedback_volatility;
    float hurst_exponent;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
