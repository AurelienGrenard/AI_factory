// rough_heston athena_autocall sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_heston/price_gradients/path_product_graph.cuh"
#include "product/athena_autocall/price_gradients/device_preparation.cuh"
#include "product/athena_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_heston {

using AthenaAutocallPriceGradientPlan = PathProductPriceGradientPlan<
    product::athena_autocall::price_gradients::DevicePreparation
>;

inline AthenaAutocallPriceGradientPlan
prepare_rough_heston_athena_autocall_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::athena_autocall::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::athena_autocall::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using AthenaAutocallNodeGraph = PathProductNodeGraph<
    product::athena_autocall::price_gradients::DevicePreparation,
    product::AthenaAutocallPathPolicy,
    volterra::RegularHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_heston
