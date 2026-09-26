// Generated merton straddle sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/terminal_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/merton/price_gradients/device_preparation.cuh"
#include "product/straddle/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::merton {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::merton::price_gradients;
namespace straddle_pg =
    ::ai_factory::workbench::product::straddle::price_gradients;

using StraddlePriceGradientPlan = epg::TerminalDevicePreparedPlan<
    mpg::DevicePreparation,
    straddle_pg::DevicePreparation
>;

void prepare_straddle_price_gradient_stencils_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_straddle_diagonal_sensitivity_stencils_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline StraddlePriceGradientPlan
prepare_merton_straddle_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::StraddleParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_terminal_device_sensitivities<
        StraddlePriceGradientPlan
    >(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline StraddlePriceGradientPlan
prepare_merton_straddle_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::StraddleParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_merton_straddle_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}


void launch_merton_straddle_price_gradients_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<pg::SensitivityOrders Orders>
void launch_merton_straddle_diagonal_sensitivities_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for terminal node-graph sensitivities.
template<pg::SensitivityOrders Orders>
std::size_t merton_straddle_node_graph_workspace_bytes(
    const StraddlePriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<pg::SensitivityOrders Orders>
void launch_merton_straddle_node_graph_sensitivities_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::merton
