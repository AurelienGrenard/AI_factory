// CIR++/Svensson zero_coupon_bond_option analytical sensitivities.
#pragma once

#include "common/fixed_income/price_gradients/device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/fixed_income/cir_plus_plus/price_gradients/device_preparation.cuh"
#include "product/zero_coupon_bond_option/price_gradients/device_preparation.cuh"
#include "curve/svensson/price_gradients/device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::svensson {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace fipg = ::ai_factory::workbench::fixed_income::price_gradients;
namespace model_pg =
    ::ai_factory::workbench::model::fixed_income::cir_plus_plus::price_gradients;
namespace zero_coupon_bond_option_pg =
    ::ai_factory::workbench::product::zero_coupon_bond_option::price_gradients;
namespace curve_pg =
    ::ai_factory::workbench::curve::svensson::price_gradients;
using CurveParameters = curve::svensson::SvenssonParameters;

using ZeroCouponBondOptionPriceGradientPlan = fipg::CurveDevicePreparedPlan<
    model_pg::DevicePreparation,
    curve_pg::DevicePreparation,
    zero_coupon_bond_option_pg::DevicePreparation
>;

void prepare_zero_coupon_bond_option_price_gradient_stencils_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_zero_coupon_bond_option_diagonal_sensitivity_stencils_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline ZeroCouponBondOptionPriceGradientPlan
prepare_cir_plus_plus_svensson_zero_coupon_bond_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const curve::svensson::SvenssonParameters> curves,
    std::span<const product::ZeroCouponBondOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return fipg::prepare_curve_device_sensitivities<ZeroCouponBondOptionPriceGradientPlan>(
        models,
        curves,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline ZeroCouponBondOptionPriceGradientPlan
prepare_cir_plus_plus_svensson_zero_coupon_bond_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const curve::svensson::SvenssonParameters> curves,
    std::span<const product::ZeroCouponBondOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_cir_plus_plus_svensson_zero_coupon_bond_option_sensitivities(
        models,
        curves,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_cir_plus_plus_svensson_zero_coupon_bond_option_price_gradients_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_cir_plus_plus_svensson_zero_coupon_bond_option_diagonal_sensitivities_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::svensson
