// rough_heston Asian-option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_heston/price_gradients/path_product_graph.cuh"
#include "product/asian_option/price_gradients/device_preparation.cuh"
#include "product/asian_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_heston {

using AsianOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::asian_option::price_gradients::DevicePreparation
>;

inline AsianOptionPriceGradientPlan
prepare_rough_heston_asian_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::AsianOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::asian_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using AsianOptionNodeGraph = PathProductNodeGraph<
    product::asian_option::price_gradients::DevicePreparation,
    product::AsianOptionPathPolicy<Side>,
    volterra::DenseHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_heston
