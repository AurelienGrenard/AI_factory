// Generated variance_gamma exact-terminal straddle sensitivities over the common device-prepared engine.
#include "model/equity/markovian/variance_gamma/product/straddle_price_gradients.cuh"

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/terminal_node_graph_launcher.cuh"
#include "common/equity/price_gradients/terminal_product_sensitivity_policy.cuh"
#include "model/equity/markovian/variance_gamma/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/variance_gamma/product/straddle.cuh"
#include "product/straddle/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::variance_gamma {

void prepare_straddle_price_gradient_stencils_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "variance_gamma.straddle.price_gradients.stencil_preparation"
    );
}

void prepare_straddle_diagonal_sensitivity_stencils_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "variance_gamma.straddle.diagonal.stencil_preparation"
    );
}


void launch_variance_gamma_straddle_price_gradients_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_variance_gamma_straddle_cuda(
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
            host.time.dt * static_cast<float>(
                host.time.simulation_steps_per_day
            ),
            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_first_sensitivities<
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::StraddlePathPolicy>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "variance_gamma.straddle.price_gradients.device_prepared",
        "default/nodes=3/B=1"
    );
}

template<pg::SensitivityOrders Orders>
void launch_variance_gamma_straddle_diagonal_sensitivities_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_variance_gamma_straddle_cuda(
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
            host.time.dt * static_cast<float>(
                host.time.simulation_steps_per_day
            ),
            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_diagonal_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::StraddlePathPolicy>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "variance_gamma.straddle.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<pg::SensitivityOrders Orders>
std::size_t variance_gamma_straddle_node_graph_workspace_bytes(
    const StraddlePriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::terminal_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::StraddlePathPolicy>,
        8U,
        16U,
        2U
    >(host, configuration);
}

template<pg::SensitivityOrders Orders>
void launch_variance_gamma_straddle_node_graph_sensitivities_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_variance_gamma_straddle_cuda(
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
            host.time.dt * static_cast<float>(
                host.time.simulation_steps_per_day
            ),
            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_node_graph_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::StraddlePathPolicy>,
        8U,
        16U,
        2U
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        workspace,
        workspace_bytes,
        launch_price_only,
        "variance_gamma.straddle.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}

template void launch_variance_gamma_straddle_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const StraddlePriceGradientPlan&,
    StraddlePriceGradientPlan::DeviceInputs,
    StraddlePriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
variance_gamma_straddle_node_graph_workspace_bytes<pg::SensitivityOrders::second>(
    const StraddlePriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_variance_gamma_straddle_node_graph_sensitivities_cuda<pg::SensitivityOrders::second>(
    const StraddlePriceGradientPlan&,
    StraddlePriceGradientPlan::DeviceInputs,
    StraddlePriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_variance_gamma_straddle_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const StraddlePriceGradientPlan&,
    StraddlePriceGradientPlan::DeviceInputs,
    StraddlePriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
variance_gamma_straddle_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>(
    const StraddlePriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_variance_gamma_straddle_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const StraddlePriceGradientPlan&,
    StraddlePriceGradientPlan::DeviceInputs,
    StraddlePriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::variance_gamma
