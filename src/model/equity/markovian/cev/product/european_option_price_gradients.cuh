// Generated cev European price/selected-gradient launcher and host scenario builder.
#pragma once
#include "common/price_gradients/launch.cuh"
#include "common/option_side.cuh"
#include "model/equity/markovian/cev/price_gradients/parameter_policy.cuh"
#include "product/european_option/price_gradients/parameter_policy.cuh"

namespace ai_factory::workbench::model::equity::cev {
using EuropeanOptionPriceGradientPlan = ::ai_factory::workbench::equity::price_gradients::ScenarioPlan<
    ModelParameters, product::EuropeanOptionParameters>;

inline EuropeanOptionPriceGradientPlan prepare_cev_european_option_price_gradients(
    std::span<const ModelParameters> models, std::span<const product::EuropeanOptionParameters> products,
    PriceConstruction construction, ::ai_factory::workbench::equity::price_gradients::TimeConfiguration time,
    const ::ai_factory::workbench::price_gradients::PriceGradientConfiguration& configuration
) {
    return ::ai_factory::workbench::equity::price_gradients::prepare_scenarios<
        price_gradients::ParameterPolicy, product::european_option::price_gradients::ParameterPolicy>(
            models, products, construction, time, configuration);
}

template<OptionSide Side>
void launch_cev_european_option_price_gradients_cuda(
    const EuropeanOptionPriceGradientPlan& host,
    ::ai_factory::workbench::price_gradients::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType> device,
    const ::ai_factory::workbench::price_gradients::LaunchConfiguration& configuration,
    ::ai_factory::workbench::price_gradients::Outputs outputs
);
}  // namespace ai_factory::workbench::model::equity::cev
