// Generated heston_3_2 geometric_asian_option sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/heston_3_2/price_gradients/device_preparation.cuh"
#include "product/geometric_asian_option/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::heston_3_2 {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::heston_3_2::price_gradients;
namespace geometric_asian_option_pg =
    ::ai_factory::workbench::product::geometric_asian_option::price_gradients;

using GeometricAsianOptionPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    geometric_asian_option_pg::DevicePreparation
>;

void prepare_geometric_asian_option_price_gradient_stencils_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_geometric_asian_option_diagonal_sensitivity_stencils_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline GeometricAsianOptionPriceGradientPlan
prepare_heston_3_2_geometric_asian_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::GeometricAsianOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        GeometricAsianOptionPriceGradientPlan,
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

inline GeometricAsianOptionPriceGradientPlan
prepare_heston_3_2_geometric_asian_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::GeometricAsianOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_heston_3_2_geometric_asian_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_heston_3_2_geometric_asian_option_price_gradients_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_heston_3_2_geometric_asian_option_diagonal_sensitivities_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t heston_3_2_geometric_asian_option_node_graph_workspace_bytes(
    const GeometricAsianOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_heston_3_2_geometric_asian_option_node_graph_sensitivities_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


// Selected mixed derivatives use the node graph; request shape lives in host.
template<OptionSide Side>
std::size_t heston_3_2_geometric_asian_option_mixed_node_graph_workspace_bytes(
    const GeometricAsianOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side>
void launch_heston_3_2_geometric_asian_option_mixed_node_graph_sensitivities_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    GeometricAsianOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::heston_3_2
