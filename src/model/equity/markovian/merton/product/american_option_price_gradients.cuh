// Generated merton American sensitivities with row-local device preparation.
#pragma once

#include "common/longstaff_schwartz/launch.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/merton/price_gradients/device_preparation.cuh"
#include "product/american_option/price_gradients/device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::model::equity::merton {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace lspg =
    ::ai_factory::workbench::longstaff_schwartz::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::merton::price_gradients;
namespace american_option_pg =
    ::ai_factory::workbench::product::american_option::price_gradients;

using AmericanOptionPriceGradientPlan = lspg::DevicePreparedPlan<
    mpg::DevicePreparation,
    american_option_pg::DevicePreparation
>;

void prepare_american_option_price_gradient_stencils_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_american_option_diagonal_sensitivity_stencils_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline AmericanOptionPriceGradientPlan
prepare_merton_american_option_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::AmericanOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return lspg::prepare_device_sensitivities<
        AmericanOptionPriceGradientPlan
    >(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline AmericanOptionPriceGradientPlan
prepare_merton_american_option_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::AmericanOptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_merton_american_option_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_merton_american_option_price_gradients_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_merton_american_option_diagonal_sensitivities_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t merton_american_option_node_graph_workspace_bytes(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_merton_american_option_node_graph_sensitivities_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


template<OptionSide Side>
std::size_t merton_american_option_mixed_node_graph_workspace_bytes(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_merton_american_option_mixed_node_graph_sensitivities_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    AmericanOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);

}  // namespace ai_factory::workbench::model::equity::merton
