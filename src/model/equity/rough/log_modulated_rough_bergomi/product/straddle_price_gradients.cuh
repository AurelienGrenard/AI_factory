// log_modulated_rough_bergomi straddle sensitivities through the terminal rough graph.
#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/price_gradients/terminal_product_graph.cuh"
#include "product/straddle/price_gradients/device_preparation.cuh"
#include "product/straddle/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

namespace straddle_pg =
    ::ai_factory::workbench::product::straddle::price_gradients;

using StraddlePriceGradientPlan =
    TerminalProductPriceGradientPlan<
        straddle_pg::DevicePreparation
    >;

inline StraddlePriceGradientPlan
prepare_log_modulated_rough_bergomi_straddle_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::StraddleParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_terminal_product_sensitivities<
        straddle_pg::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using StraddleNodeGraph = TerminalProductNodeGraph<
    straddle_pg::DevicePreparation,
    product::StraddlePathPolicy,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
