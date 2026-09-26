// Generated normal_inverse_gaussian asset_or_nothing_option sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/terminal_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/price_gradients/device_preparation.cuh"
#include "product/asset_or_nothing_option/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::normal_inverse_gaussian::price_gradients;
namespace asset_or_nothing_option_pg =
    ::ai_factory::workbench::product::asset_or_nothing_option::price_gradients;

using AssetOrNothingOptionPriceGradientPlan = epg::TerminalDevicePreparedPlan<
    mpg::DevicePreparation,
    asset_or_nothing_option_pg::DevicePreparation
>;

void prepare_asset_or_nothing_option_price_gradient_stencils_cuda(
    const AssetOrNothingOptionPriceGradientPlan& host,
    AssetOrNothingOptionPriceGradientPlan::DeviceInputs device,
    AssetOrNothingOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_asset_or_nothing_option_diagonal_sensitivity_stencils_cuda(
    const AssetOrNothingOptionPriceGradientPlan& host,
    AssetOrNothingOptionPriceGradientPlan::DeviceInputs device,
    AssetOrNothingOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline AssetOrNothingOptionPriceGradientPlan
prepare_normal_inverse_gaussian_asset_or_nothing_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::AssetOrNothingOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_terminal_device_sensitivities<
        AssetOrNothingOptionPriceGradientPlan
    >(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline AssetOrNothingOptionPriceGradientPlan
prepare_normal_inverse_gaussian_asset_or_nothing_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::AssetOrNothingOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_normal_inverse_gaussian_asset_or_nothing_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_normal_inverse_gaussian_asset_or_nothing_option_price_gradients_cuda(
    const AssetOrNothingOptionPriceGradientPlan& host,
    AssetOrNothingOptionPriceGradientPlan::DeviceInputs device,
    AssetOrNothingOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_normal_inverse_gaussian_asset_or_nothing_option_diagonal_sensitivities_cuda(
    const AssetOrNothingOptionPriceGradientPlan& host,
    AssetOrNothingOptionPriceGradientPlan::DeviceInputs device,
    AssetOrNothingOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for terminal node-graph sensitivities.
template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t normal_inverse_gaussian_asset_or_nothing_option_node_graph_workspace_bytes(
    const AssetOrNothingOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_normal_inverse_gaussian_asset_or_nothing_option_node_graph_sensitivities_cuda(
    const AssetOrNothingOptionPriceGradientPlan& host,
    AssetOrNothingOptionPriceGradientPlan::DeviceInputs device,
    AssetOrNothingOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);

// Selected mixed derivatives use the node graph; request shape lives in host.
template<OptionSide Side>
std::size_t normal_inverse_gaussian_asset_or_nothing_option_mixed_node_graph_workspace_bytes(
    const AssetOrNothingOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side>
void launch_normal_inverse_gaussian_asset_or_nothing_option_mixed_node_graph_sensitivities_cuda(
    const AssetOrNothingOptionPriceGradientPlan& host,
    AssetOrNothingOptionPriceGradientPlan::DeviceInputs device,
    AssetOrNothingOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    AssetOrNothingOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
