// Generated kou up_no_touch sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/kou/price_gradients/device_preparation.cuh"
#include "product/up_no_touch/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::kou {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::kou::price_gradients;
namespace up_no_touch_pg =
    ::ai_factory::workbench::product::up_no_touch::price_gradients;

using UpNoTouchPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    up_no_touch_pg::DevicePreparation
>;

void prepare_up_no_touch_price_gradient_stencils_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_up_no_touch_diagonal_sensitivity_stencils_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline UpNoTouchPriceGradientPlan
prepare_kou_up_no_touch_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::UpNoTouchParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        UpNoTouchPriceGradientPlan,
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

inline UpNoTouchPriceGradientPlan
prepare_kou_up_no_touch_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::UpNoTouchParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_kou_up_no_touch_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}


void launch_kou_up_no_touch_price_gradients_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<pg::SensitivityOrders Orders>
void launch_kou_up_no_touch_diagonal_sensitivities_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<pg::SensitivityOrders Orders>
std::size_t kou_up_no_touch_node_graph_workspace_bytes(
    const UpNoTouchPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<pg::SensitivityOrders Orders>
void launch_kou_up_no_touch_node_graph_sensitivities_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::kou
