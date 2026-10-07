// log_modulated_rough_bergomi cliquet sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/price_gradients/path_product_graph.cuh"
#include "product/cliquet/price_gradients/device_preparation.cuh"
#include "product/cliquet/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

using CliquetPriceGradientPlan = PathProductPriceGradientPlan<
    product::cliquet::price_gradients::DevicePreparation
>;

inline CliquetPriceGradientPlan
prepare_log_modulated_rough_bergomi_cliquet_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::cliquet::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::cliquet::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using CliquetNodeGraph = PathProductNodeGraph<
    product::cliquet::price_gradients::DevicePreparation,
    product::CliquetPathPolicy,
    volterra::RegularHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
