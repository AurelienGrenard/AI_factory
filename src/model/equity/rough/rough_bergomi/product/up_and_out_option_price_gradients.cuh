// rough_bergomi up_and_out_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_bergomi/price_gradients/path_product_graph.cuh"
#include "product/up_and_out_option/price_gradients/device_preparation.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_bergomi {

using UpAndOutOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::up_and_out_option::price_gradients::DevicePreparation
>;

inline UpAndOutOptionPriceGradientPlan
prepare_rough_bergomi_up_and_out_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::up_and_out_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::up_and_out_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using UpAndOutOptionNodeGraph = PathProductNodeGraph<
    product::up_and_out_option::price_gradients::DevicePreparation,
    product::UpAndOutOptionPathPolicy<Side>,
    volterra::DenseHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
