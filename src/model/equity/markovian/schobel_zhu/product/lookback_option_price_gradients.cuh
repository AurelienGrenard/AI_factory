// Generated schobel_zhu lookback_option sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/schobel_zhu/price_gradients/device_preparation.cuh"
#include "product/lookback_option/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::schobel_zhu {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::schobel_zhu::price_gradients;
namespace lookback_option_pg =
    ::ai_factory::workbench::product::lookback_option::price_gradients;

using LookbackOptionPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    lookback_option_pg::DevicePreparation
>;

void prepare_lookback_option_price_gradient_stencils_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_lookback_option_diagonal_sensitivity_stencils_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline LookbackOptionPriceGradientPlan
prepare_schobel_zhu_lookback_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::LookbackOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        LookbackOptionPriceGradientPlan,
        false
    >(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline LookbackOptionPriceGradientPlan
prepare_schobel_zhu_lookback_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::LookbackOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_schobel_zhu_lookback_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}


void launch_schobel_zhu_lookback_option_price_gradients_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<pg::SensitivityOrders Orders>
void launch_schobel_zhu_lookback_option_diagonal_sensitivities_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<pg::SensitivityOrders Orders>
std::size_t schobel_zhu_lookback_option_node_graph_workspace_bytes(
    const LookbackOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<pg::SensitivityOrders Orders>
void launch_schobel_zhu_lookback_option_node_graph_sensitivities_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


// Selected mixed derivatives use the node graph; request shape lives in host.

std::size_t schobel_zhu_lookback_option_mixed_node_graph_workspace_bytes(
    const LookbackOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);


void launch_schobel_zhu_lookback_option_mixed_node_graph_sensitivities_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    LookbackOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::schobel_zhu
