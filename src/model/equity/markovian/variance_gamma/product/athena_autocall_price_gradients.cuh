// Generated variance_gamma athena_autocall sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/path_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/variance_gamma/price_gradients/device_preparation.cuh"
#include "product/athena_autocall/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::variance_gamma {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::variance_gamma::price_gradients;
namespace athena_autocall_pg =
    ::ai_factory::workbench::product::athena_autocall::price_gradients;

using AthenaAutocallPriceGradientPlan = epg::PathDevicePreparedPlan<
    mpg::DevicePreparation,
    athena_autocall_pg::DevicePreparation
>;

void prepare_athena_autocall_price_gradient_stencils_cuda(
    const AthenaAutocallPriceGradientPlan& host,
    AthenaAutocallPriceGradientPlan::DeviceInputs device,
    AthenaAutocallPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_athena_autocall_diagonal_sensitivity_stencils_cuda(
    const AthenaAutocallPriceGradientPlan& host,
    AthenaAutocallPriceGradientPlan::DeviceInputs device,
    AthenaAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline AthenaAutocallPriceGradientPlan
prepare_variance_gamma_athena_autocall_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::AthenaAutocallParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_path_device_sensitivities<
        AthenaAutocallPriceGradientPlan,
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

inline AthenaAutocallPriceGradientPlan
prepare_variance_gamma_athena_autocall_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::AthenaAutocallParameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_variance_gamma_athena_autocall_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}


void launch_variance_gamma_athena_autocall_price_gradients_cuda(
    const AthenaAutocallPriceGradientPlan& host,
    AthenaAutocallPriceGradientPlan::DeviceInputs device,
    AthenaAutocallPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<pg::SensitivityOrders Orders>
void launch_variance_gamma_athena_autocall_diagonal_sensitivities_cuda(
    const AthenaAutocallPriceGradientPlan& host,
    AthenaAutocallPriceGradientPlan::DeviceInputs device,
    AthenaAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

// Public workspace and launch surface for path node-graph sensitivities.
template<pg::SensitivityOrders Orders>
std::size_t variance_gamma_athena_autocall_node_graph_workspace_bytes(
    const AthenaAutocallPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<pg::SensitivityOrders Orders>
void launch_variance_gamma_athena_autocall_node_graph_sensitivities_cuda(
    const AthenaAutocallPriceGradientPlan& host,
    AthenaAutocallPriceGradientPlan::DeviceInputs device,
    AthenaAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


}  // namespace ai_factory::workbench::model::equity::variance_gamma
