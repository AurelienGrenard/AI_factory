// rough_sabr up_and_in_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_sabr/price_gradients/path_product_graph.cuh"
#include "product/up_and_in_option/price_gradients/device_preparation.cuh"
#include "product/up_and_in_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_sabr {

using UpAndInOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::up_and_in_option::price_gradients::DevicePreparation
>;

inline UpAndInOptionPriceGradientPlan
prepare_rough_sabr_up_and_in_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::up_and_in_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::up_and_in_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using UpAndInOptionNodeGraph = PathProductNodeGraph<
    product::up_and_in_option::price_gradients::DevicePreparation,
    product::UpAndInOptionPathPolicy<Side>,
    volterra::DenseHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_sabr
