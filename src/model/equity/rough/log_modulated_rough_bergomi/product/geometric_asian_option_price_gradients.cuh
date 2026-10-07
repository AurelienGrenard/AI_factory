// log_modulated_rough_bergomi geometric_asian_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/price_gradients/path_product_graph.cuh"
#include "product/geometric_asian_option/price_gradients/device_preparation.cuh"
#include "product/geometric_asian_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

using GeometricAsianOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::geometric_asian_option::price_gradients::DevicePreparation
>;

inline GeometricAsianOptionPriceGradientPlan
prepare_log_modulated_rough_bergomi_geometric_asian_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::geometric_asian_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::geometric_asian_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using GeometricAsianOptionNodeGraph = PathProductNodeGraph<
    product::geometric_asian_option::price_gradients::DevicePreparation,
    product::GeometricAsianOptionPathPolicy<Side>,
    volterra::DenseHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
