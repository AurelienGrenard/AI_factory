// Generated black_scholes cliquet sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/black_scholes/price_gradients/device_preparation.cuh"
#include "product/cliquet/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::black_scholes {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::black_scholes::price_gradients;
namespace cliquet_pg =
    ::ai_factory::workbench::product::cliquet::price_gradients;

using CliquetPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    cliquet_pg::DevicePreparation
>;

void prepare_cliquet_price_gradient_stencils_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_cliquet_diagonal_sensitivity_stencils_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline CliquetPriceGradientPlan
prepare_black_scholes_cliquet_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::CliquetParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        CliquetPriceGradientPlan,
        true
    >(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline CliquetPriceGradientPlan
prepare_black_scholes_cliquet_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::CliquetParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_black_scholes_cliquet_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}


void launch_black_scholes_cliquet_price_gradients_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<pg::SensitivityOrders Orders>
void launch_black_scholes_cliquet_diagonal_sensitivities_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<pg::SensitivityOrders Orders>
std::size_t black_scholes_cliquet_node_graph_workspace_bytes(
    const CliquetPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<pg::SensitivityOrders Orders>
void launch_black_scholes_cliquet_node_graph_sensitivities_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


// Selected mixed derivatives use the node graph; request shape lives in host.

std::size_t black_scholes_cliquet_mixed_node_graph_workspace_bytes(
    const CliquetPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);


void launch_black_scholes_cliquet_mixed_node_graph_sensitivities_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    CliquetPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::black_scholes
