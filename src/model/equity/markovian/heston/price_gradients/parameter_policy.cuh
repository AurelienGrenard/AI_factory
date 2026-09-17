// Heston sensitivity parameter policy; validation does not impose the Feller condition.
#pragma once
#include "common/equity/price_gradients/scenarios.hpp"
#include "model/equity/markovian/heston/parameters.hpp"

namespace ai_factory::workbench::model::equity::heston::price_gradients {
struct ParameterPolicy {
    using Parameters = ModelParameters;
    static constexpr bool kMultiplicativeSpot = true;
    using Field = ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>;
    static constexpr std::array fields{
        Field{"model.spot", &Parameters::spot}, Field{"model.risk_free_rate", &Parameters::risk_free_rate},
        Field{"model.dividend_yield", &Parameters::dividend_yield}, Field{"model.initial_variance", &Parameters::initial_variance},
        Field{"model.kappa", &Parameters::kappa}, Field{"model.theta", &Parameters::theta},
        Field{"model.gamma", &Parameters::gamma}, Field{"model.rho", &Parameters::rho}
    };
    static bool valid(const Parameters& p) {
        for (const auto& field : fields) if (!std::isfinite(p.*field.member)) return false;
        return p.spot > 0 && p.initial_variance >= 0 && p.kappa > 0 && p.theta > 0
            && p.gamma > 0 && p.rho >= -1 && p.rho <= 1;
    }
};
}  // namespace ai_factory::workbench::model::equity::heston::price_gradients
