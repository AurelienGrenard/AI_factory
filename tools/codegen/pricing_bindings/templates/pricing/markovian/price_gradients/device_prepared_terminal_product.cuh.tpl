// Generated ${model} ${product} sensitivities with row-local device preparation.
#pragma once

#include "common/equity/price_gradients/terminal_device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
${mixed_output_include}#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/equity/markovian/${model}/price_gradients/device_preparation.cuh"
#include "product/${product}/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <span>

namespace ai_factory::workbench::model::equity::${model} {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::equity::${model}::price_gradients;
namespace ${product}_pg =
    ::ai_factory::workbench::product::${product}::price_gradients;

using ${product_type}PriceGradientPlan = epg::TerminalDevicePreparedPlan<
    mpg::DevicePreparation,
    ${product}_pg::DevicePreparation
>;

void prepare_${product}_price_gradient_stencils_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

void prepare_${product}_diagonal_sensitivity_stencils_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
);

inline ${product_type}PriceGradientPlan
prepare_${model}_${product}_sensitivities(
    std::span<const ModelParameters> models,
    std::span<const product::${product_type}Parameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_terminal_device_sensitivities<
        ${product_type}PriceGradientPlan
    >(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

inline ${product_type}PriceGradientPlan
prepare_${model}_${product}_price_gradients(
    std::span<const ModelParameters> models,
    std::span<const product::${product_type}Parameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_${model}_${product}_sensitivities(
        models,
        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

${side_template}
void launch_${model}_${product}_price_gradients_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

${sensitivity_template}
void launch_${model}_${product}_diagonal_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

${node_graph_declarations}

}  // namespace ai_factory::workbench::model::equity::${model}
