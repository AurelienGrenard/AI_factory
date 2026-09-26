// Generated G2++/Svensson Bermudan sensitivities.
#pragma once

#include "common/fixed_income/price_gradients/device_prepared_plan.hpp"
#include "common/fixed_income/swaption_side.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "curve/svensson/price_gradients/device_preparation.cuh"
#include "model/fixed_income/g2_plus_plus/price_gradients/device_preparation.cuh"
#include "product/bermudan_swaption/price_gradients/device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace fipg = ::ai_factory::workbench::fixed_income::price_gradients;
namespace model_pg =
    ::ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients;
namespace bermudan_pg =
    ::ai_factory::workbench::product::bermudan_swaption::price_gradients;
namespace curve_pg =
    ::ai_factory::workbench::curve::svensson::price_gradients;

using BermudanSwaptionPriceGradientPlan = fipg::CurveDevicePreparedPlan<
    model_pg::DevicePreparation,
    curve_pg::DevicePreparation,
    bermudan_pg::DevicePreparation
>;

void prepare_bermudan_swaption_price_gradient_stencils_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_bermudan_swaption_diagonal_sensitivity_stencils_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline BermudanSwaptionPriceGradientPlan
prepare_g2_plus_plus_svensson_bermudan_swaption_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const curve::svensson::SvenssonParameters> curves,
    std::span<const product::BermudanSwaptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return fipg::prepare_curve_device_sensitivities<BermudanSwaptionPriceGradientPlan>(
        models,
        curves,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline BermudanSwaptionPriceGradientPlan
prepare_g2_plus_plus_svensson_bermudan_swaption_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const curve::svensson::SvenssonParameters> curves,
    std::span<const product::BermudanSwaptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_g2_plus_plus_svensson_bermudan_swaption_sensitivities(
        models,
        curves,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_g2_plus_plus_svensson_bermudan_swaption_price_gradients_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_g2_plus_plus_svensson_bermudan_swaption_diagonal_sensitivities_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

template<SwaptionSide Side, pg::SensitivityOrders Orders>
std::size_t g2_plus_plus_svensson_bermudan_swaption_node_graph_workspace_bytes(
    const BermudanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_g2_plus_plus_svensson_bermudan_swaption_node_graph_sensitivities_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


template<SwaptionSide Side>
std::size_t g2_plus_plus_svensson_bermudan_swaption_mixed_node_graph_workspace_bytes(
    const BermudanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_g2_plus_plus_svensson_bermudan_swaption_mixed_node_graph_sensitivities_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    BermudanSwaptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson
