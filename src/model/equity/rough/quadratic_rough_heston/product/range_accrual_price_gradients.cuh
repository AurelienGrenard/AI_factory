// quadratic_rough_heston range_accrual sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/price_gradients/path_product_graph.cuh"
#include "product/range_accrual/price_gradients/device_preparation.cuh"
#include "product/range_accrual/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

using RangeAccrualPriceGradientPlan = PathProductPriceGradientPlan<
    product::range_accrual::price_gradients::DevicePreparation
>;

inline RangeAccrualPriceGradientPlan
prepare_quadratic_rough_heston_range_accrual_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::range_accrual::price_gradients::DevicePreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_path_product_sensitivities<
        product::range_accrual::price_gradients::DevicePreparation
    >(models, products, construction, time, configuration, request);
}

template<
    std::size_t FactorCount,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using RangeAccrualNodeGraph = PathProductNodeGraph<
    product::range_accrual::price_gradients::DevicePreparation,
    product::RangeAccrualPathPolicy,
    volterra::RegularHybridSchedule,
    FactorCount, Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
