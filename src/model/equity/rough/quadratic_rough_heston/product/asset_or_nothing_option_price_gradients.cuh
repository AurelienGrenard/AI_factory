// quadratic_rough_heston asset_or_nothing_option sensitivities through the terminal rough graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/price_gradients/terminal_product_graph.cuh"
#include "product/asset_or_nothing_option/price_gradients/device_preparation.cuh"
#include "product/asset_or_nothing_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

namespace asset_or_nothing_option_pg =
    ::ai_factory::workbench::product::asset_or_nothing_option::price_gradients;

using AssetOrNothingOptionPriceGradientPlan =
    TerminalProductPriceGradientPlan<
        asset_or_nothing_option_pg::DevicePreparation
    >;

inline AssetOrNothingOptionPriceGradientPlan
prepare_quadratic_rough_heston_asset_or_nothing_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::AssetOrNothingOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_terminal_product_sensitivities<
        asset_or_nothing_option_pg::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using AssetOrNothingOptionNodeGraph = TerminalProductNodeGraph<
    asset_or_nothing_option_pg::DevicePreparation,
    product::AssetOrNothingOptionPathPolicy<Side>,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
