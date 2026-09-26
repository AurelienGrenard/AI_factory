// Generated schobel_zhu asian_option sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/schobel_zhu/price_gradients/device_preparation.cuh"
#include "product/asian_option/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::schobel_zhu {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::schobel_zhu::price_gradients;
namespace asian_option_pg =
    ::ai_factory::workbench::product::asian_option::price_gradients;

using AsianOptionPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    asian_option_pg::DevicePreparation
>;

void prepare_asian_option_price_gradient_stencils_cuda(
    const AsianOptionPriceGradientPlan& host,
    AsianOptionPriceGradientPlan::DeviceInputs device,
    AsianOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_asian_option_diagonal_sensitivity_stencils_cuda(
    const AsianOptionPriceGradientPlan& host,
    AsianOptionPriceGradientPlan::DeviceInputs device,
    AsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline AsianOptionPriceGradientPlan
prepare_schobel_zhu_asian_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::AsianOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        AsianOptionPriceGradientPlan,
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

inline AsianOptionPriceGradientPlan
prepare_schobel_zhu_asian_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::AsianOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_schobel_zhu_asian_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_schobel_zhu_asian_option_price_gradients_cuda(
    const AsianOptionPriceGradientPlan& host,
    AsianOptionPriceGradientPlan::DeviceInputs device,
    AsianOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_schobel_zhu_asian_option_diagonal_sensitivities_cuda(
    const AsianOptionPriceGradientPlan& host,
    AsianOptionPriceGradientPlan::DeviceInputs device,
    AsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t schobel_zhu_asian_option_node_graph_workspace_bytes(
    const AsianOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_schobel_zhu_asian_option_node_graph_sensitivities_cuda(
    const AsianOptionPriceGradientPlan& host,
    AsianOptionPriceGradientPlan::DeviceInputs device,
    AsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::schobel_zhu
