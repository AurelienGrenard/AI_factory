// rough_stein_stein lookback_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_stein_stein/price_gradients/path_product_graph.cuh"
#include "product/lookback_option/price_gradients/device_preparation.cuh"
#include "product/lookback_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_stein_stein {

using LookbackOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::lookback_option::price_gradients::DevicePreparation
>;

inline LookbackOptionPriceGradientPlan
prepare_rough_stein_stein_lookback_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::lookback_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::lookback_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using LookbackOptionNodeGraph = PathProductNodeGraph<
    product::lookback_option::price_gradients::DevicePreparation,
    product::LookbackOptionPathPolicy,
    volterra::DenseHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
