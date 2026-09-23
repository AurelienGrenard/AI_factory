// Compose existing lognormal analytics with explicit scenario maturity in years.
#pragma once
#include "common/equity/price_gradients/scenarios.hpp"
#include "product/european_option/closed_form_pricing_policy.cuh"

namespace ai_factory::workbench::product::european_option::price_gradients {
template<typename Model, OptionSide Side>
struct ClosedFormPolicy {
    using InputRow = equity::price_gradients::Scenario<Model, EuropeanOptionParameters>;
    using Base = LognormalEuropeanOptionClosedFormPolicy<Model, Side>;
    __device__ static float evaluate(const InputRow& input) {
        const auto prepared = prepare_vanilla_option_values(prepare_analytics(input.model),
            input.product.strike, input.maturity_years);
        return Base::evaluate_price(prepared);
    }
};
}  // namespace ai_factory::workbench::product::european_option::price_gradients
