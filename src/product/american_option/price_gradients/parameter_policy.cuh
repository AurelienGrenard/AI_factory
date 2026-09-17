// American-option sensitivity coordinates; the exercise calendar stays fixed.
#pragma once

#include "common/equity/price_gradients/scenarios.hpp"
#include "product/american_option/parameters.hpp"

#include <array>
#include <cmath>

namespace ai_factory::workbench::product::american_option::price_gradients {

struct ParameterPolicy {
    using Parameters = AmericanOptionParameters;
    using Field = equity::price_gradients::ParameterField<Parameters>;

    static constexpr bool kSupportsMaturitySensitivity = false;
    static constexpr std::array fields{
        Field{"product.strike", &Parameters::strike},
    };

    static bool valid(const Parameters& parameters) {
        return std::isfinite(parameters.strike)
            && parameters.strike > 0.0f
            && parameters.maturity_days > 0U
            && parameters.exercise_interval_days > 0U
            && parameters.exercise_interval_days < parameters.maturity_days;
    }
};

}  // namespace ai_factory::workbench::product::american_option::price_gradients
