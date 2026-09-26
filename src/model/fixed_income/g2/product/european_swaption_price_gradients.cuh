// G2 European-swaption MC sensitivities.
#pragma once

#include "common/fixed_income/price_gradients/device_prepared_plan.hpp"
#include "common/fixed_income/swaption_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/fixed_income/g2/price_gradients/device_preparation.cuh"
#include "product/european_swaption/price_gradients/device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::model::fixed_income::g2 {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace fipg = ::ai_factory::workbench::fixed_income::price_gradients;
namespace model_pg =
    ::ai_factory::workbench::model::fixed_income::g2::price_gradients;
namespace european_swaption_pg =
    ::ai_factory::workbench::product::european_swaption::price_gradients;

using EuropeanSwaptionPriceGradientPlan = fipg::DevicePreparedPlan<
    model_pg::DevicePreparation,
    european_swaption_pg::DevicePreparation
>;

void prepare_european_swaption_price_gradient_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_european_swaption_diagonal_sensitivity_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline EuropeanSwaptionPriceGradientPlan
prepare_g2_european_swaption_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::RegularEuropeanSwaptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return fipg::prepare_device_sensitivities<EuropeanSwaptionPriceGradientPlan>(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline EuropeanSwaptionPriceGradientPlan
prepare_g2_european_swaption_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::RegularEuropeanSwaptionParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_g2_european_swaption_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<SwaptionSide Side>
void launch_g2_european_swaption_price_gradients_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<SwaptionSide Side, pg::SensitivityOrders Orders>
void launch_g2_european_swaption_diagonal_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

template<SwaptionSide Side, pg::SensitivityOrders Orders>
std::size_t g2_european_swaption_node_graph_workspace_bytes(
    const EuropeanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<SwaptionSide Side, pg::SensitivityOrders Orders>
void launch_g2_european_swaption_node_graph_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);

// Selected first, diagonal and mixed terms share one compact terminal graph.
template<SwaptionSide Side>
std::size_t g2_european_swaption_mixed_node_graph_workspace_bytes(
    const EuropeanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<SwaptionSide Side>
void launch_g2_european_swaption_mixed_node_graph_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    EuropeanSwaptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);

}  // namespace ai_factory::workbench::model::fixed_income::g2
