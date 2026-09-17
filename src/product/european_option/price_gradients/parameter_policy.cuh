// European strike sensitivity and validation of the unchanged input calendar.
#pragma once
#include "common/equity/price_gradients/scenarios.hpp"
#include "product/european_option/parameters.hpp"

namespace ai_factory::workbench::product::european_option::price_gradients {
struct ParameterPolicy {
    using Parameters = EuropeanOptionParameters;
    static constexpr std::array fields{
        ::ai_factory::workbench::equity::price_gradients::ParameterField<Parameters>{"product.strike", &Parameters::strike}
    };
    static bool valid(const Parameters& p) {
        return std::isfinite(p.strike) && p.strike > 0 && p.maturity_days > 0U;
    }
};
}  // namespace ai_factory::workbench::product::european_option::price_gradients
