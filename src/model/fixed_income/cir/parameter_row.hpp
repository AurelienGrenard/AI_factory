// Shared parameter-row parsing and domain checks for a CIR factor (standalone or shifted).
#pragma once

#include "model/fixed_income/cir/parameter_domain.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace ai_factory::workbench::model::fixed_income::cir {
inline ModelParameters parse_parameter_row(
    const nlohmann::json& parameters, const std::string& prefix
) {
    const ModelParameters model = {
        {
            parameters.at("mean_reversion").get<float>(),
            parameters.at("long_term_mean").get<float>(),
            parameters.at("volatility").get<float>(),
        },
        parameters.at("initial_state").get<float>(),
    };
    validate_parameters(model, prefix);
    return model;
}
}  // namespace ai_factory::workbench::model::fixed_income::cir
