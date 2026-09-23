// Merton selected-gradient fields. Intensity and maturity are excluded here.
#pragma once
#include "common/equity/price_gradients/scenarios.hpp"
#include "model/equity/markovian/merton/parameters.hpp"

namespace ai_factory::workbench::model::equity::merton::price_gradients {
struct ParameterPolicy {
    using Parameters = ModelParameters;
    using Field = ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = false;
    static constexpr std::array fields{
        Field{"model.spot", &Parameters::spot},
        Field{"model.risk_free_rate", &Parameters::risk_free_rate},
        Field{"model.dividend_yield", &Parameters::dividend_yield},
        Field{"model.volatility", &Parameters::volatility},
        Field{"model.jump_log_mean", &Parameters::jump_log_mean},
        Field{"model.jump_log_volatility", &Parameters::jump_log_volatility}
    };
    static bool valid(const Parameters& p) {
        return std::isfinite(p.spot) && p.spot > 0 && std::isfinite(p.risk_free_rate)
            && std::isfinite(p.dividend_yield) && std::isfinite(p.volatility) && p.volatility > 0
            && std::isfinite(p.jump_intensity) && p.jump_intensity >= 0
            && std::isfinite(p.jump_log_mean) && std::isfinite(p.jump_log_volatility)
            && p.jump_log_volatility >= 0;
    }
};
}  // namespace ai_factory::workbench::model::equity::merton::price_gradients
