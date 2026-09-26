// Generated heston fixed-step digital_option sensitivities over the common device-prepared engine.
#include "model/equity/markovian/heston/product/digital_option_price_gradients.cuh"

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/mixed_terminal_node_graph_launcher.cuh"
#include "common/equity/price_gradients/terminal_node_graph_launcher.cuh"
#include "common/equity/price_gradients/terminal_product_sensitivity_policy.cuh"
#include "model/equity/markovian/heston/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/heston/product/digital_option.cuh"
#include "product/digital_option/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::heston {

void prepare_digital_option_price_gradient_stencils_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "heston.digital_option.price_gradients.stencil_preparation"
    );
}

void prepare_digital_option_diagonal_sensitivity_stencils_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "heston.digital_option.diagonal.stencil_preparation"
    );
}

template<OptionSide Side>
void launch_heston_digital_option_price_gradients_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_heston_digital_option_cuda<Side>(
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
            host.time.dt,
            host.time.simulation_steps_per_day,
            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_first_sensitivities<
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "heston.digital_option.price_gradients.device_prepared",
        Side == OptionSide::call ? "call/nodes=3/B=1" : "put/nodes=3/B=1"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_heston_digital_option_diagonal_sensitivities_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_heston_digital_option_cuda<Side>(
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
            host.time.dt,
            host.time.simulation_steps_per_day,
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
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "heston.digital_option.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t heston_digital_option_node_graph_workspace_bytes(
    const DigitalOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::terminal_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>,
        11U,
        32U,
        2U
    >(host, configuration);
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_heston_digital_option_node_graph_sensitivities_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_heston_digital_option_cuda<Side>(
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
            host.time.dt,
            host.time.simulation_steps_per_day,
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
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>,
        11U,
        32U,
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
        "heston.digital_option.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}

template<OptionSide Side>
std::size_t heston_digital_option_mixed_node_graph_workspace_bytes(
    const DigitalOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::mixed_terminal_node_graph_workspace_bytes<
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>,
        11U,
        55U,
        128U,
        2U
    >(host, configuration);
}

template<OptionSide Side>
void launch_heston_digital_option_mixed_node_graph_sensitivities_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    DigitalOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    epg::launch_mixed_terminal_node_graph_sensitivities<
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>,
        11U,
        55U,
        128U,
        2U
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
        "heston.digital_option.sensitivities.mixed_node_graph",
        "selected_gradient_and_hessian/nodes=graph"
    );
}

template void launch_heston_digital_option_price_gradients_cuda<OptionSide::call>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::StencilOutputs,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_heston_digital_option_diagonal_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_digital_option_node_graph_workspace_bytes<OptionSide::call, pg::SensitivityOrders::second>(
    const DigitalOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_digital_option_node_graph_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_heston_digital_option_diagonal_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::first_and_second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_digital_option_node_graph_workspace_bytes<OptionSide::call, pg::SensitivityOrders::first_and_second>(
    const DigitalOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_digital_option_node_graph_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::first_and_second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template std::size_t
heston_digital_option_mixed_node_graph_workspace_bytes<OptionSide::call>(
    const DigitalOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_digital_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    DigitalOptionPriceGradientPlan::MixedStencilOutputs,
    const pg::LaunchConfiguration&,
    pg::SensitivityOutputs, pg::MixedSensitivityOutputs,
    void*, std::size_t);

template void launch_heston_digital_option_price_gradients_cuda<OptionSide::put>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::StencilOutputs,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_heston_digital_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_digital_option_node_graph_workspace_bytes<OptionSide::put, pg::SensitivityOrders::second>(
    const DigitalOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_digital_option_node_graph_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_heston_digital_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_digital_option_node_graph_workspace_bytes<OptionSide::put, pg::SensitivityOrders::first_and_second>(
    const DigitalOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_digital_option_node_graph_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template std::size_t
heston_digital_option_mixed_node_graph_workspace_bytes<OptionSide::put>(
    const DigitalOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_digital_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>(
    const DigitalOptionPriceGradientPlan&,
    DigitalOptionPriceGradientPlan::DeviceInputs,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs,
    DigitalOptionPriceGradientPlan::MixedStencilOutputs,
    const pg::LaunchConfiguration&,
    pg::SensitivityOutputs, pg::MixedSensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::heston
