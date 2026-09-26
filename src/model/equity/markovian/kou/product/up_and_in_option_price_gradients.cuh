// Generated kou up_and_in_option sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/kou/price_gradients/device_preparation.cuh"
#include "product/up_and_in_option/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::kou {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::kou::price_gradients;
namespace up_and_in_option_pg =
    ::ai_factory::workbench::product::up_and_in_option::price_gradients;

using UpAndInOptionPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    up_and_in_option_pg::DevicePreparation
>;

void prepare_up_and_in_option_price_gradient_stencils_cuda(
    const UpAndInOptionPriceGradientPlan& host,
    UpAndInOptionPriceGradientPlan::DeviceInputs device,
    UpAndInOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_up_and_in_option_diagonal_sensitivity_stencils_cuda(
    const UpAndInOptionPriceGradientPlan& host,
    UpAndInOptionPriceGradientPlan::DeviceInputs device,
    UpAndInOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline UpAndInOptionPriceGradientPlan
prepare_kou_up_and_in_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::UpAndInOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        UpAndInOptionPriceGradientPlan,
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

inline UpAndInOptionPriceGradientPlan
prepare_kou_up_and_in_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::UpAndInOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_kou_up_and_in_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_kou_up_and_in_option_price_gradients_cuda(
    const UpAndInOptionPriceGradientPlan& host,
    UpAndInOptionPriceGradientPlan::DeviceInputs device,
    UpAndInOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_kou_up_and_in_option_diagonal_sensitivities_cuda(
    const UpAndInOptionPriceGradientPlan& host,
    UpAndInOptionPriceGradientPlan::DeviceInputs device,
    UpAndInOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t kou_up_and_in_option_node_graph_workspace_bytes(
    const UpAndInOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_kou_up_and_in_option_node_graph_sensitivities_cuda(
    const UpAndInOptionPriceGradientPlan& host,
    UpAndInOptionPriceGradientPlan::DeviceInputs device,
    UpAndInOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::kou
