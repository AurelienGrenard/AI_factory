// rough_stein_stein phoenix_memory_autocall sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_stein_stein/price_gradients/path_product_graph.cuh"
#include "product/phoenix_memory_autocall/price_gradients/device_preparation.cuh"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_stein_stein {

using PhoenixMemoryAutocallPriceGradientPlan = PathProductPriceGradientPlan<
    product::phoenix_memory_autocall::price_gradients::DevicePreparation
>;

inline PhoenixMemoryAutocallPriceGradientPlan
prepare_rough_stein_stein_phoenix_memory_autocall_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::phoenix_memory_autocall::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::phoenix_memory_autocall::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using PhoenixMemoryAutocallNodeGraph = PathProductNodeGraph<
    product::phoenix_memory_autocall::price_gradients::DevicePreparation,
    product::PhoenixMemoryAutocallPathPolicy,
    volterra::RegularHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
