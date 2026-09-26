// Generated sabr digital_option sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/terminal_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/sabr/price_gradients/device_preparation.cuh"
#include "product/digital_option/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::sabr {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::sabr::price_gradients;
namespace digital_option_pg =
    ::ai_factory::workbench::product::digital_option::price_gradients;

using DigitalOptionPriceGradientPlan = epg::TerminalDevicePreparedPlan<
    mpg::DevicePreparation,
    digital_option_pg::DevicePreparation
>;

void prepare_digital_option_price_gradient_stencils_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_digital_option_diagonal_sensitivity_stencils_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline DigitalOptionPriceGradientPlan
prepare_sabr_digital_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::DigitalOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_terminal_device_sensitivities<
        DigitalOptionPriceGradientPlan
    >(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline DigitalOptionPriceGradientPlan
prepare_sabr_digital_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::DigitalOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_sabr_digital_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_sabr_digital_option_price_gradients_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_sabr_digital_option_diagonal_sensitivities_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for terminal node-graph sensitivities.
template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t sabr_digital_option_node_graph_workspace_bytes(
    const DigitalOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_sabr_digital_option_node_graph_sensitivities_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);

// Selected mixed derivatives use the node graph; request shape lives in host.
template<OptionSide Side>
std::size_t sabr_digital_option_mixed_node_graph_workspace_bytes(
    const DigitalOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side>
void launch_sabr_digital_option_mixed_node_graph_sensitivities_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    DigitalOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::sabr
