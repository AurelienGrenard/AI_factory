// rough_heston cliquet sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_heston/price_gradients/path_product_graph.cuh"
#include "product/cliquet/price_gradients/device_preparation.cuh"
#include "product/cliquet/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_heston {

using CliquetPriceGradientPlan = PathProductPriceGradientPlan<
    product::cliquet::price_gradients::DevicePreparation
>;

inline CliquetPriceGradientPlan
prepare_rough_heston_cliquet_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::cliquet::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::cliquet::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using CliquetNodeGraph = PathProductNodeGraph<
    product::cliquet::price_gradients::DevicePreparation,
    product::CliquetPathPolicy,
    volterra::RegularHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_heston
