// rough_stein_stein up_one_touch sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_stein_stein/price_gradients/path_product_graph.cuh"
#include "product/up_one_touch/price_gradients/device_preparation.cuh"
#include "product/up_one_touch/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_stein_stein {

using UpOneTouchPriceGradientPlan = PathProductPriceGradientPlan<
    product::up_one_touch::price_gradients::DevicePreparation
>;

inline UpOneTouchPriceGradientPlan
prepare_rough_stein_stein_up_one_touch_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::up_one_touch::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::up_one_touch::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using UpOneTouchNodeGraph = PathProductNodeGraph<
    product::up_one_touch::price_gradients::DevicePreparation,
    product::UpOneTouchPathPolicy,
    volterra::DenseHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
