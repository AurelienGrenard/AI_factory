// Generated merton phoenix_autocall sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/merton/price_gradients/device_preparation.cuh"
#include "product/phoenix_autocall/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::merton {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::merton::price_gradients;
namespace phoenix_autocall_pg =
    ::ai_factory::workbench::product::phoenix_autocall::price_gradients;

using PhoenixAutocallPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    phoenix_autocall_pg::DevicePreparation
>;

void prepare_phoenix_autocall_price_gradient_stencils_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_phoenix_autocall_diagonal_sensitivity_stencils_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline PhoenixAutocallPriceGradientPlan
prepare_merton_phoenix_autocall_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::PhoenixAutocallParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        PhoenixAutocallPriceGradientPlan,
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

inline PhoenixAutocallPriceGradientPlan
prepare_merton_phoenix_autocall_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::PhoenixAutocallParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_merton_phoenix_autocall_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}


void launch_merton_phoenix_autocall_price_gradients_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<pg::SensitivityOrders Orders>
void launch_merton_phoenix_autocall_diagonal_sensitivities_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<pg::SensitivityOrders Orders>
std::size_t merton_phoenix_autocall_node_graph_workspace_bytes(
    const PhoenixAutocallPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<pg::SensitivityOrders Orders>
void launch_merton_phoenix_autocall_node_graph_sensitivities_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::merton
