// quadratic_rough_heston phoenix_memory_autocall sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/price_gradients/path_product_graph.cuh"
#include "product/phoenix_memory_autocall/price_gradients/device_preparation.cuh"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

using PhoenixMemoryAutocallPriceGradientPlan = PathProductPriceGradientPlan<
    product::phoenix_memory_autocall::price_gradients::DevicePreparation
>;

inline PhoenixMemoryAutocallPriceGradientPlan
prepare_quadratic_rough_heston_phoenix_memory_autocall_sensitivities(
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
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using PhoenixMemoryAutocallNodeGraph = PathProductNodeGraph<
    product::phoenix_memory_autocall::price_gradients::DevicePreparation,
    product::PhoenixMemoryAutocallPathPolicy,
    volterra::RegularHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
