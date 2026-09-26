// ${model_display}${curve_display_suffix} ${product} analytical sensitivities.
#include "${unit_path}.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/closed_form/price_gradients/device_prepared_mixed_kernel.cuh"
#include "common/fixed_income/price_gradients/closed_form_policy.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "product/${product}/pricing_policy.cuh"
${implementation_include}
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::${binding_namespace} {

void prepare_${product}_price_gradient_stencils_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host, device, stencil_outputs, result_offset, result_count,
        "${diagnostic_name}.sensitivities.prepare_stencils"
    );
}

void prepare_${product}_diagonal_sensitivity_stencils_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "${diagnostic_name}.sensitivities.prepare_diagonal_stencils"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders, typename Outputs>
void launch_sensitivities(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ::ai_factory::workbench::monte_carlo::price_gradients::
        DevicePreparedStencilOutputs<
            pg::SensitivityTraits<Orders>::node_capacity
        > stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    Outputs outputs
) {
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "${diagnostic_name} sensitivities require closed-form pricing."
        );
    }
    using PricingPolicy = ${pricing_policy};
    using Policy = ${scenario_policy};
    closed_form::price_gradients::launch_device_prepared<Orders, Policy>(
        host,
        device,
        stencil_outputs,
        configuration,
        pg::as_sensitivity_outputs(outputs),
        "${diagnostic_name}.sensitivities.closed_form",
        option_side_name(Side)
    );
}

template<OptionSide Side>
void launch_${function_prefix}_${product}_price_gradients_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    launch_sensitivities<Side, pg::SensitivityOrders::first>(
        host, device, stencil_outputs, configuration, outputs
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_${function_prefix}_${product}_diagonal_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_sensitivities<Side, Orders>(
        host, device, stencil_outputs, configuration, outputs
    );
}

template<OptionSide Side>
std::size_t ${function_prefix}_${product}_mixed_node_graph_workspace_bytes(
    const ${product_type}PriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    (void)configuration;
    return closed_form::price_gradients::mixed_workspace_bytes(host);
}

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
) {
    using PricingPolicy = ${pricing_policy};
    using Policy = ${scenario_policy};
    closed_form::price_gradients::launch_device_prepared_mixed<
        Policy,
        ${maximum}U,
        ${mixed_maximum}U
    >(
        host,
        device,
        stencil_outputs,
        mixed_stencil_outputs,
        configuration,
        outputs,
        mixed_outputs,
        workspace,
        workspace_bytes,
        "${diagnostic_name}.sensitivities.closed_form_mixed",
        option_side_name(Side)
    );
}

#define AI_FACTORY_INSTANTIATE(side)                                        \
template void launch_${function_prefix}_${product}_price_gradients_cuda<    \
    side>(const ${product_type}PriceGradientPlan&,                           \
    ${product_type}PriceGradientPlan::DeviceInputs,                          \
    ${product_type}PriceGradientPlan::StencilOutputs,                        \
    const pg::LaunchConfiguration&, pg::Outputs);                            \
template void launch_${function_prefix}_${product}_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::second>(                                   \
    const ${product_type}PriceGradientPlan&,                                 \
    ${product_type}PriceGradientPlan::DeviceInputs,                          \
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs,                \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);                 \
template void launch_${function_prefix}_${product}_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::first_and_second>(                         \
    const ${product_type}PriceGradientPlan&,                                 \
    ${product_type}PriceGradientPlan::DeviceInputs,                          \
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs,                \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);             \
template std::size_t                                                       \
${function_prefix}_${product}_mixed_node_graph_workspace_bytes<side>(       \
    const ${product_type}PriceGradientPlan&,                                \
    const pg::LaunchConfiguration&);                                        \
template void                                                              \
launch_${function_prefix}_${product}_mixed_node_graph_sensitivities_cuda<side>( \
    const ${product_type}PriceGradientPlan&,                                \
    ${product_type}PriceGradientPlan::DeviceInputs,                         \
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs,               \
    ${product_type}PriceGradientPlan::MixedStencilOutputs,                  \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,                 \
    pg::MixedSensitivityOutputs, void*, std::size_t)

AI_FACTORY_INSTANTIATE(OptionSide::call);
AI_FACTORY_INSTANTIATE(OptionSide::put);
#undef AI_FACTORY_INSTANTIATE

}  // namespace ai_factory::workbench::model::fixed_income::${binding_namespace}
