// rough_heston phoenix_autocall sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_heston/price_gradients/path_product_graph.cuh"
#include "product/phoenix_autocall/price_gradients/device_preparation.cuh"
#include "product/phoenix_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_heston {

using PhoenixAutocallPriceGradientPlan = PathProductPriceGradientPlan<
    product::phoenix_autocall::price_gradients::DevicePreparation
>;

inline PhoenixAutocallPriceGradientPlan
prepare_rough_heston_phoenix_autocall_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::phoenix_autocall::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::phoenix_autocall::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using PhoenixAutocallNodeGraph = PathProductNodeGraph<
    product::phoenix_autocall::price_gradients::DevicePreparation,
    product::PhoenixAutocallPathPolicy,
    volterra::RegularHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_heston
