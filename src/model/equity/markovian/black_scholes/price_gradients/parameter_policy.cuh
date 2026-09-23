// Black-Scholes sensitivity parameter policy and complete finite model domain.
#pragma once
#include "common/equity/price_gradients/scenarios.hpp"
#include "model/equity/markovian/black_scholes/parameters.hpp"

namespace ai_factory::workbench::model::equity::black_scholes::price_gradients {
struct ParameterPolicy {
    using Parameters = ModelParameters;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr std::array fields{
        ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>{"model.spot", &Parameters::spot},
        ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>{"model.risk_free_rate", &Parameters::risk_free_rate},
        ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>{"model.dividend_yield", &Parameters::dividend_yield},
        ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>{"model.volatility", &Parameters::volatility}
    };
    static bool valid(const Parameters& p) {
        return std::isfinite(p.spot) && p.spot > 0 && std::isfinite(p.risk_free_rate)
            && std::isfinite(p.dividend_yield) && std::isfinite(p.volatility) && p.volatility > 0;
    }
};
}  // namespace ai_factory::workbench::model::equity::black_scholes::price_gradients
