// Hull-White/Nelson-Siegel zero_coupon_bond_option analytical sensitivities.
#pragma once

#include "common/fixed_income/price_gradients/device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/fixed_income/hull_white/price_gradients/device_preparation.cuh"
#include "product/zero_coupon_bond_option/price_gradients/device_preparation.cuh"
#include "curve/nelson_siegel/price_gradients/device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace fipg = ::ai_factory::workbench::fixed_income::price_gradients;
namespace model_pg =
    ::ai_factory::workbench::model::fixed_income::hull_white::price_gradients;
namespace zero_coupon_bond_option_pg =
    ::ai_factory::workbench::product::zero_coupon_bond_option::price_gradients;
namespace curve_pg =
    ::ai_factory::workbench::curve::nelson_siegel::price_gradients;
using CurveParameters = curve::nelson_siegel::NelsonSiegelParameters;

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
prepare_hull_white_nelson_siegel_zero_coupon_bond_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const curve::nelson_siegel::NelsonSiegelParameters> curves,
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
prepare_hull_white_nelson_siegel_zero_coupon_bond_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const curve::nelson_siegel::NelsonSiegelParameters> curves,
    std::span<const product::ZeroCouponBondOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_hull_white_nelson_siegel_zero_coupon_bond_option_sensitivities(
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
void launch_hull_white_nelson_siegel_zero_coupon_bond_option_price_gradients_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_hull_white_nelson_siegel_zero_coupon_bond_option_diagonal_sensitivities_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

template<OptionSide Side>
std::size_t hull_white_nelson_siegel_zero_coupon_bond_option_mixed_node_graph_workspace_bytes(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side>
void launch_hull_white_nelson_siegel_zero_coupon_bond_option_mixed_node_graph_sensitivities_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    ZeroCouponBondOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);

}  // namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel
