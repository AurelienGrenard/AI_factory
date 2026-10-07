// rough_sabr range_accrual sensitivities through the shared rough path graph.
#pragma once

#include "model/equity/rough/rough_sabr/price_gradients/path_product_graph.cuh"
#include "product/range_accrual/price_gradients/device_preparation.cuh"
#include "product/range_accrual/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_sabr {

using RangeAccrualPriceGradientPlan = PathProductPriceGradientPlan<
    product::range_accrual::price_gradients::DevicePreparation
>;

inline RangeAccrualPriceGradientPlan
prepare_rough_sabr_range_accrual_sensitivities(
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
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities = 16U>
using RangeAccrualNodeGraph = PathProductNodeGraph<
    product::range_accrual::price_gradients::DevicePreparation,
    product::RangeAccrualPathPolicy,
    volterra::RegularHybridSchedule,
    Orders, MaximumSensitivities
>;

}  // namespace ai_factory::workbench::model::equity::rough_sabr
