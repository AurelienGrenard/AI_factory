// Heston American price and selected gradients under frozen central exercise.
#pragma once

#include "common/longstaff_schwartz/launch.cuh"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "model/equity/markovian/heston/price_gradients/parameter_policy.cuh"
#include "product/american_option/price_gradients/parameter_policy.cuh"

#include <span>

namespace ai_factory::workbench::model::equity::heston {

using AmericanOptionPriceGradientPlan =
    ::ai_factory::workbench::equity::price_gradients::ScenarioPlan<
        ModelParameters,
        product::AmericanOptionParameters
    >;

inline AmericanOptionPriceGradientPlan
prepare_heston_american_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::AmericanOptionParameters> products,
    PriceConstruction construction,
    ::ai_factory::workbench::equity::price_gradients::TimeConfiguration time,
    const ::ai_factory::workbench::price_gradients::
        PriceGradientConfiguration& configuration
) {
    return ::ai_factory::workbench::equity::price_gradients::prepare_scenarios<
        price_gradients::ParameterPolicy,
        product::american_option::price_gradients::ParameterPolicy
    >(models, products, construction, time, configuration);
}

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_heston_american_option_price_gradients_cuda(
    const AmericanOptionPriceGradientPlan& host,
    ::ai_factory::workbench::price_gradients::DeviceInputs<
        AmericanOptionPriceGradientPlan::ScenarioType
    > device,
    const ::ai_factory::workbench::price_gradients::LaunchConfiguration&
        configuration,
    ::ai_factory::workbench::price_gradients::Outputs outputs
);

}  // namespace ai_factory::workbench::model::equity::heston
