// rough_sabr double_knock_out_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_sabr/price_gradients/path_product_graph.cuh"
#include "product/double_knock_out_option/price_gradients/device_preparation.cuh"
#include "product/double_knock_out_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_sabr {

using DoubleKnockOutOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::double_knock_out_option::price_gradients::DevicePreparation
>;

inline DoubleKnockOutOptionPriceGradientPlan
prepare_rough_sabr_double_knock_out_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::double_knock_out_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::double_knock_out_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using DoubleKnockOutOptionNodeGraph = PathProductNodeGraph<
    product::double_knock_out_option::price_gradients::DevicePreparation,
    product::DoubleKnockOutOptionPathPolicy<Side>,
    volterra::DenseHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_sabr
