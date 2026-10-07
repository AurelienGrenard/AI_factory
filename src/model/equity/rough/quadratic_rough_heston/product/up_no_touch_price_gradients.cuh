// quadratic_rough_heston up_no_touch sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/price_gradients/path_product_graph.cuh"
#include "product/up_no_touch/price_gradients/device_preparation.cuh"
#include "product/up_no_touch/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

using UpNoTouchPriceGradientPlan = PathProductPriceGradientPlan<
    product::up_no_touch::price_gradients::DevicePreparation
>;

inline UpNoTouchPriceGradientPlan
prepare_quadratic_rough_heston_up_no_touch_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::up_no_touch::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::up_no_touch::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using UpNoTouchNodeGraph = PathProductNodeGraph<
    product::up_no_touch::price_gradients::DevicePreparation,
    product::UpNoTouchPathPolicy,
    volterra::DenseHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
