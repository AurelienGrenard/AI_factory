// rough_sabr forward_start_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_sabr/price_gradients/path_product_graph.cuh"
#include "product/forward_start_option/price_gradients/device_preparation.cuh"
#include "product/forward_start_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_sabr {

using ForwardStartOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::forward_start_option::price_gradients::DevicePreparation
>;

inline ForwardStartOptionPriceGradientPlan
prepare_rough_sabr_forward_start_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::forward_start_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::forward_start_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using ForwardStartOptionNodeGraph = PathProductNodeGraph<
    product::forward_start_option::price_gradients::DevicePreparation,
    product::ForwardStartOptionPathPolicy<Side>,
    volterra::CalendarHybridSchedule<2U>,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_sabr
