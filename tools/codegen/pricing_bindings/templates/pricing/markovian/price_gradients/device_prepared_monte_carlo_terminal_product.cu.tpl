// Generated ${model} ${transition_description} ${product} sensitivities over the common device-prepared engine.
#include "model/equity/markovian/${model}/product/${product}_price_gradients.cuh"

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/terminal_node_graph_launcher.cuh"
#include "common/equity/price_gradients/terminal_product_sensitivity_policy.cuh"
#include "model/equity/markovian/${model}/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/${model}/product/${product}.cuh"
#include "${product_policy_header}"

#include <algorithm>

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

${side_template}
void launch_${model}_${product}_price_gradients_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_${model}_${product}_cuda${side_argument}(
            device.models,
            host.models.size(),
            host.products.data(),
            device.products,
            host.products.size(),
            host.construction,
            host.result_count,
            configuration.result_offset,
            configuration.result_count,
            configuration.paths_per_price,
${price_only_time_arguments}            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_first_sensitivities<
        mpg::CoupledDynamics,
        ${sensitivity_policy}
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "${model}.${product}.price_gradients.device_prepared",
        ${first_variant}
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
    const auto launch_price_only = [&] {
        launch_${model}_${product}_cuda${side_argument}(
            device.models,
            host.models.size(),
            host.products.data(),
            device.products,
            host.products.size(),
            host.construction,
            host.result_count,
            configuration.result_offset,
            configuration.result_count,
            configuration.paths_per_price,
${price_only_time_arguments}            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_diagonal_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        ${sensitivity_policy}
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "${model}.${product}.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

${sensitivity_template}
std::size_t ${model}_${product}_node_graph_workspace_bytes(
    const ${product_type}PriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::terminal_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        ${sensitivity_policy},
        ${maximum}U,
        ${graph_group_size}U,
        ${graph_nodes_per_worker}U
    >(host, configuration);
}

${sensitivity_template}
void launch_${model}_${product}_node_graph_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_${model}_${product}_cuda${side_argument}(
            device.models,
            host.models.size(),
            host.products.data(),
            device.products,
            host.products.size(),
            host.construction,
            host.result_count,
            configuration.result_offset,
            configuration.result_count,
            configuration.paths_per_price,
${price_only_time_arguments}            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_node_graph_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        ${sensitivity_policy},
        ${maximum}U,
        ${graph_group_size}U,
        ${graph_nodes_per_worker}U
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        workspace,
        workspace_bytes,
        launch_price_only,
        "${model}.${product}.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}

${explicit_instantiations}

}  // namespace ai_factory::workbench::model::equity::${model}
