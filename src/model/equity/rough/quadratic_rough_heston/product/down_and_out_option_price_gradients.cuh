// quadratic_rough_heston down_and_out_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/price_gradients/path_product_graph.cuh"
#include "product/down_and_out_option/price_gradients/device_preparation.cuh"
#include "product/down_and_out_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

using DownAndOutOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::down_and_out_option::price_gradients::DevicePreparation
>;

inline DownAndOutOptionPriceGradientPlan
prepare_quadratic_rough_heston_down_and_out_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::down_and_out_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::down_and_out_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using DownAndOutOptionNodeGraph = PathProductNodeGraph<
    product::down_and_out_option::price_gradients::DevicePreparation,
    product::DownAndOutOptionPathPolicy<Side>,
    volterra::DenseHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
