// Generated cev forward_start_option sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/cev/price_gradients/device_preparation.cuh"
#include "product/forward_start_option/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::cev {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::cev::price_gradients;
namespace forward_start_option_pg =
    ::ai_factory::workbench::product::forward_start_option::price_gradients;

using ForwardStartOptionPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    forward_start_option_pg::DevicePreparation
>;

void prepare_forward_start_option_price_gradient_stencils_cuda(
    const ForwardStartOptionPriceGradientPlan& host,
    ForwardStartOptionPriceGradientPlan::DeviceInputs device,
    ForwardStartOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_forward_start_option_diagonal_sensitivity_stencils_cuda(
    const ForwardStartOptionPriceGradientPlan& host,
    ForwardStartOptionPriceGradientPlan::DeviceInputs device,
    ForwardStartOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline ForwardStartOptionPriceGradientPlan
prepare_cev_forward_start_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::ForwardStartOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        ForwardStartOptionPriceGradientPlan,
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

inline ForwardStartOptionPriceGradientPlan
prepare_cev_forward_start_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::ForwardStartOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_cev_forward_start_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_cev_forward_start_option_price_gradients_cuda(
    const ForwardStartOptionPriceGradientPlan& host,
    ForwardStartOptionPriceGradientPlan::DeviceInputs device,
    ForwardStartOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_cev_forward_start_option_diagonal_sensitivities_cuda(
    const ForwardStartOptionPriceGradientPlan& host,
    ForwardStartOptionPriceGradientPlan::DeviceInputs device,
    ForwardStartOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t cev_forward_start_option_node_graph_workspace_bytes(
    const ForwardStartOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_cev_forward_start_option_node_graph_sensitivities_cuda(
    const ForwardStartOptionPriceGradientPlan& host,
    ForwardStartOptionPriceGradientPlan::DeviceInputs device,
    ForwardStartOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::cev
