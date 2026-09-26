// ${model_display}${curve_display_suffix} ${product} analytical sensitivities.
#pragma once

#include "common/fixed_income/price_gradients/device_prepared_plan.hpp"
#include "common/option_side.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "model/fixed_income/${model}/price_gradients/device_preparation.cuh"
#include "product/${product}/price_gradients/device_preparation.cuh"
${curve_header_include}
#include <span>

namespace ai_factory::workbench::model::fixed_income::${binding_namespace} {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace fipg = ::ai_factory::workbench::fixed_income::price_gradients;
namespace model_pg =
    ::ai_factory::workbench::model::fixed_income::${model}::price_gradients;
namespace ${product_namespace}_pg =
    ::ai_factory::workbench::product::${product_namespace}::price_gradients;
${curve_namespace_alias}${curve_type_alias}
using ${product_type}PriceGradientPlan = ${plan_type};

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
prepare_${function_prefix}_${product}_sensitivities(
    std::span<const ModelParameters> models,
${curve_span_argument}    std::span<const product::${product_type}Parameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return ${prepare_function}<${product_type}PriceGradientPlan>(
        models,
${curve_prepare_argument}        products,
        construction,
        time,
        configuration,
        request
    );
}

inline ${product_type}PriceGradientPlan
prepare_${function_prefix}_${product}_price_gradients(
    std::span<const ModelParameters> models,
${curve_span_argument}    std::span<const product::${product_type}Parameters> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    return prepare_${function_prefix}_${product}_sensitivities(
        models,
${curve_prepare_argument}        products,
        construction,
        time,
        configuration,
        {pg::SensitivityOrders::first}
    );
}

template<OptionSide Side>
void launch_${function_prefix}_${product}_price_gradients_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
);

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_${function_prefix}_${product}_diagonal_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
);

template<OptionSide Side>
std::size_t ${function_prefix}_${product}_mixed_node_graph_workspace_bytes(
    const ${product_type}PriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

template<OptionSide Side>
void launch_${function_prefix}_${product}_mixed_node_graph_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    ${product_type}PriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);

}  // namespace ai_factory::workbench::model::fixed_income::${binding_namespace}
