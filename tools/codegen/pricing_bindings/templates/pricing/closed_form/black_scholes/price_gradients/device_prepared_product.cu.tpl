// Generated Black-Scholes closed-form ${product} sensitivities.
#include "model/equity/markovian/${model}/product/${product}_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/closed_form/price_gradients/device_prepared_mixed_kernel.cuh"
#include "common/equity/price_gradients/scenario_closed_form_policy.cuh"
#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "model/equity/markovian/${model}/product/${product}_impl.cuh"

#include <stdexcept>

namespace ai_factory::workbench::model::equity::${model} {

void prepare_${product}_price_gradient_stencils_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "${model}.${product}.price_gradients.stencil_preparation"
    );
}

void prepare_${product}_diagonal_sensitivity_stencils_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "${model}.${product}.diagonal.stencil_preparation"
    );
}

${closed_form_helper_template}
void launch_${product}_closed_form_sensitivities(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    using ScenarioClosedFormPolicy =
        epg::ScenarioClosedFormPolicy<${closed_form_policy}>;
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "${model} ${product} sensitivities require closed-form pricing."
        );
    }
    closed_form::price_gradients::launch_device_prepared<
        Orders,
        ScenarioClosedFormPolicy
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        "${model}.${product}.sensitivities.closed_form",
        Orders == pg::SensitivityOrders::first
            ? "gradient/nodes=3"
            : Orders == pg::SensitivityOrders::second
                ? "diagonal_hessian/nodes=4"
                : "gradient_and_diagonal_hessian/nodes=4"
    );
}

${side_template}
void launch_${model}_${product}_price_gradients_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    launch_${product}_closed_form_sensitivities<
        ${closed_form_first_helper_arguments}
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        pg::as_sensitivity_outputs(outputs)
    );
}

${sensitivity_template}
void launch_${model}_${product}_diagonal_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_${product}_closed_form_sensitivities<
        ${closed_form_diagonal_helper_arguments}
    >(
        host, device, stencil_outputs, configuration, outputs
    );
}

${side_template}
std::size_t ${model}_${product}_mixed_node_graph_workspace_bytes(
    const ${product_type}PriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    (void)configuration;
    return closed_form::price_gradients::mixed_workspace_bytes(host);
}

${side_template}
void launch_${model}_${product}_mixed_node_graph_sensitivities_cuda(
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
    using ScenarioClosedFormPolicy =
        epg::ScenarioClosedFormPolicy<${closed_form_policy}>;
    closed_form::price_gradients::launch_device_prepared_mixed<
        ScenarioClosedFormPolicy,
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
        "${model}.${product}.sensitivities.closed_form_mixed",
        "selected_gradient_and_hessian"
    );
}

${closed_form_explicit_instantiations}

}  // namespace ai_factory::workbench::model::equity::${model}
