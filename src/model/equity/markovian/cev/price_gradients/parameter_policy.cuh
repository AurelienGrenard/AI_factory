// Host CEV parameter policy for selected gradients and nonhomogeneous spot paths.
#pragma once
#include "common/equity/price_gradients/scenarios.hpp"
#include "model/equity/markovian/cev/parameters.hpp"

namespace ai_factory::workbench::model::equity::cev::price_gradients {
struct ParameterPolicy {
    using Parameters = ModelParameters;
    static constexpr bool kMultiplicativeSpot = false;
    using Field = ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>;
    static constexpr std::array fields{
        Field{"model.spot", &Parameters::spot}, Field{"model.risk_free_rate", &Parameters::risk_free_rate},
        Field{"model.dividend_yield", &Parameters::dividend_yield}, Field{"model.sigma", &Parameters::sigma},
        Field{"model.beta", &Parameters::beta}
    };
    static bool valid(const Parameters& p) {
        for (const auto& field : fields) if (!std::isfinite(p.*field.member)) return false;
        return p.spot > 0 && p.sigma > 0 && p.beta >= .5f && p.beta < 1.f;
    }
};
}  // namespace ai_factory::workbench::model::equity::cev::price_gradients
