// quadratic_rough_heston down_and_in_option sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/price_gradients/path_product_graph.cuh"
#include "product/down_and_in_option/price_gradients/device_preparation.cuh"
#include "product/down_and_in_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

using DownAndInOptionPriceGradientPlan = PathProductPriceGradientPlan<
    product::down_and_in_option::price_gradients::DevicePreparation
>;

inline DownAndInOptionPriceGradientPlan
prepare_quadratic_rough_heston_down_and_in_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::down_and_in_option::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::down_and_in_option::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    OptionSide Side,
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using DownAndInOptionNodeGraph = PathProductNodeGraph<
    product::down_and_in_option::price_gradients::DevicePreparation,
    product::DownAndInOptionPathPolicy<Side>,
    volterra::DenseHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
