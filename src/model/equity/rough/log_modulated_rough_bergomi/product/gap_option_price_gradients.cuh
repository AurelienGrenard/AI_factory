// log_modulated_rough_bergomi gap_option sensitivities through the terminal rough graph.
#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/price_gradients/terminal_product_graph.cuh"
#include "product/gap_option/price_gradients/device_preparation.cuh"
#include "product/gap_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

namespace gap_option_pg =
    ::ai_factory::workbench::product::gap_option::price_gradients;

using GapOptionPriceGradientPlan =
    TerminalProductPriceGradientPlan<
        gap_option_pg::DevicePreparation
    >;

inline GapOptionPriceGradientPlan
prepare_log_modulated_rough_bergomi_gap_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::GapOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_terminal_product_sensitivities<
        gap_option_pg::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using GapOptionNodeGraph = TerminalProductNodeGraph<
    gap_option_pg::DevicePreparation,
    product::GapOptionPathPolicy<Side>,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
