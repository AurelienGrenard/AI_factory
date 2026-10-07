// quadratic_rough_heston gap_option sensitivities through the terminal rough graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/price_gradients/terminal_product_graph.cuh"
#include "product/gap_option/price_gradients/device_preparation.cuh"
#include "product/gap_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

namespace gap_option_pg =
    ::ai_factory::workbench::product::gap_option::price_gradients;

using GapOptionPriceGradientPlan =
    TerminalProductPriceGradientPlan<
        gap_option_pg::DevicePreparation
    >;

inline GapOptionPriceGradientPlan
prepare_quadratic_rough_heston_gap_option_sensitivities(
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
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using GapOptionNodeGraph = TerminalProductNodeGraph<
    gap_option_pg::DevicePreparation,
    product::GapOptionPathPolicy<Side>,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
