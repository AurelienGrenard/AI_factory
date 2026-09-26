// Generated merton lookback_option path sensitivities over the common device-prepared engine.
#include "model/equity/markovian/merton/product/lookback_option_price_gradients.cuh"

#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/path_node_graph_launcher.cuh"
#include "common/equity/price_gradients/mixed_path_node_graph_launcher.cuh"
#include "model/equity/markovian/merton/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/merton/product/lookback_option.cuh"
#include "product/lookback_option/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::merton {

using LookbackOptionGradientSchedule = simulation::FixedStepDenseSchedule<merton::DynamicsPolicy>;

void prepare_lookback_option_price_gradient_stencils_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "merton.lookback_option.price_gradients.stencil_preparation"
    );
}

void prepare_lookback_option_diagonal_sensitivity_stencils_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "merton.lookback_option.diagonal.stencil_preparation"
    );
}


void launch_merton_lookback_option_price_gradients_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_merton_lookback_option_cuda(
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
    epg::launch_path_first_sensitivities<
        mpg::CoupledDynamics,
        product::LookbackOptionPathPolicy,
        LookbackOptionGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "merton.lookback_option.price_gradients.device_prepared",
        "default/nodes=3/B=1"
    );
}

template<pg::SensitivityOrders Orders>
void launch_merton_lookback_option_diagonal_sensitivities_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_merton_lookback_option_cuda(
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
    epg::launch_path_diagonal_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        product::LookbackOptionPathPolicy,
        LookbackOptionGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "merton.lookback_option.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<pg::SensitivityOrders Orders>
std::size_t merton_lookback_option_node_graph_workspace_bytes(
    const LookbackOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::path_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        product::LookbackOptionPathPolicy,
        LookbackOptionGradientSchedule,
        8U,
        16U,
        2U
    >(host, configuration);
}

template<pg::SensitivityOrders Orders>
void launch_merton_lookback_option_node_graph_sensitivities_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_merton_lookback_option_cuda(
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
    epg::launch_path_node_graph_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        product::LookbackOptionPathPolicy,
        LookbackOptionGradientSchedule,
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
        "merton.lookback_option.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}



std::size_t merton_lookback_option_mixed_node_graph_workspace_bytes(
    const LookbackOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::mixed_path_node_graph_workspace_bytes<
        mpg::CoupledDynamics,
        product::LookbackOptionPathPolicy,
        LookbackOptionGradientSchedule,
        8U,
        28U,
        128U,
        2U
    >(host, configuration);
}


void launch_merton_lookback_option_mixed_node_graph_sensitivities_cuda(
    const LookbackOptionPriceGradientPlan& host,
    LookbackOptionPriceGradientPlan::DeviceInputs device,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    LookbackOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    epg::launch_mixed_path_node_graph_sensitivities<
        mpg::CoupledDynamics,
        product::LookbackOptionPathPolicy,
        LookbackOptionGradientSchedule,
        8U,
        28U,
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
        "merton.lookback_option.sensitivities.mixed_node_graph",
        "full_hessian"
    );
}

template void launch_merton_lookback_option_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const LookbackOptionPriceGradientPlan&,
    LookbackOptionPriceGradientPlan::DeviceInputs,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
merton_lookback_option_node_graph_workspace_bytes<pg::SensitivityOrders::second>(
    const LookbackOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_merton_lookback_option_node_graph_sensitivities_cuda<pg::SensitivityOrders::second>(
    const LookbackOptionPriceGradientPlan&,
    LookbackOptionPriceGradientPlan::DeviceInputs,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_merton_lookback_option_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const LookbackOptionPriceGradientPlan&,
    LookbackOptionPriceGradientPlan::DeviceInputs,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
merton_lookback_option_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>(
    const LookbackOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_merton_lookback_option_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const LookbackOptionPriceGradientPlan&,
    LookbackOptionPriceGradientPlan::DeviceInputs,
    LookbackOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::merton
