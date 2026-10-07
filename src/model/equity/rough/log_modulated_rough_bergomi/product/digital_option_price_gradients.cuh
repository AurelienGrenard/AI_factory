// log_modulated_rough_bergomi digital_option sensitivities through the terminal rough graph.
#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/price_gradients/terminal_product_graph.cuh"
#include "product/digital_option/price_gradients/device_preparation.cuh"
#include "product/digital_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

namespace digital_option_pg =
    ::ai_factory::workbench::product::digital_option::price_gradients;

using DigitalOptionPriceGradientPlan =
    TerminalProductPriceGradientPlan<
        digital_option_pg::DevicePreparation
    >;

inline DigitalOptionPriceGradientPlan
prepare_log_modulated_rough_bergomi_digital_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::DigitalOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_terminal_product_sensitivities<
        digital_option_pg::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using DigitalOptionNodeGraph = TerminalProductNodeGraph<
    digital_option_pg::DevicePreparation,
    product::DigitalOptionPathPolicy<Side>,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
