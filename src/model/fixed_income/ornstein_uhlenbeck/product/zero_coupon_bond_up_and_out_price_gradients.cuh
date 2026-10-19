// OU bond-barrier first-order CRN sensitivities.
#pragma once

#include "common/fixed_income/price_gradients/device_prepared_plan.hpp"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/price_gradients/device_preparation.cuh"
#include "product/zero_coupon_bond_up_and_out/price_gradients/device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace fipg = ::ai_factory::workbench::fixed_income::price_gradients;
namespace model_pg =
    ::ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck::price_gradients;
namespace zero_coupon_bond_up_and_out_pg =
    ::ai_factory::workbench::product::zero_coupon_bond_up_and_out::price_gradients;

using ZeroCouponBondUpAndOutPriceGradientPlan = fipg::DevicePreparedPlan<
    model_pg::DevicePreparation,
    zero_coupon_bond_up_and_out_pg::DevicePreparation
>;

void prepare_zero_coupon_bond_up_and_out_price_gradient_stencils_cuda(
    const ZeroCouponBondUpAndOutPriceGradientPlan& host,
    ZeroCouponBondUpAndOutPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondUpAndOutPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline ZeroCouponBondUpAndOutPriceGradientPlan
prepare_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::ZeroCouponBondUpAndOutParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return fipg::prepare_device_sensitivities<ZeroCouponBondUpAndOutPriceGradientPlan>(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

void launch_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_price_gradients_cuda(
    const ZeroCouponBondUpAndOutPriceGradientPlan& host,
    ZeroCouponBondUpAndOutPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondUpAndOutPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
