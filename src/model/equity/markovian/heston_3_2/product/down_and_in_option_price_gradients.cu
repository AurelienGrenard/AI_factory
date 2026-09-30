// Generated heston_3_2 down_and_in_option path sensitivities over the common device-prepared engine.
#include "model/equity/markovian/heston_3_2/product/down_and_in_option_price_gradients.cuh"

#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/path_node_graph_launcher.cuh"
#include "common/equity/price_gradients/mixed_path_node_graph_launcher.cuh"
#include "model/equity/markovian/heston_3_2/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/heston_3_2/product/down_and_in_option.cuh"
#include "product/down_and_in_option/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::heston_3_2 {

using DownAndInOptionGradientSchedule = simulation::FixedStepDenseSchedule<heston_3_2::DynamicsPolicy>;

void prepare_down_and_in_option_price_gradient_stencils_cuda(
    const DownAndInOptionPriceGradientPlan& host,
    DownAndInOptionPriceGradientPlan::DeviceInputs device,
    DownAndInOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "heston_3_2.down_and_in_option.price_gradients.stencil_preparation"
    );
}

void prepare_down_and_in_option_diagonal_sensitivity_stencils_cuda(
    const DownAndInOptionPriceGradientPlan& host,
    DownAndInOptionPriceGradientPlan::DeviceInputs device,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "heston_3_2.down_and_in_option.diagonal.stencil_preparation"
    );
}

template<OptionSide Side>
void launch_heston_3_2_down_and_in_option_price_gradients_cuda(
    const DownAndInOptionPriceGradientPlan& host,
    DownAndInOptionPriceGradientPlan::DeviceInputs device,
    DownAndInOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_heston_3_2_down_and_in_option_cuda<Side>(
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
        product::DownAndInOptionPathPolicy<Side>,
        DownAndInOptionGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "heston_3_2.down_and_in_option.price_gradients.device_prepared",
        Side == OptionSide::call ? "call/nodes=3/B=1" : "put/nodes=3/B=1"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_heston_3_2_down_and_in_option_diagonal_sensitivities_cuda(
    const DownAndInOptionPriceGradientPlan& host,
    DownAndInOptionPriceGradientPlan::DeviceInputs device,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_heston_3_2_down_and_in_option_cuda<Side>(
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
        product::DownAndInOptionPathPolicy<Side>,
        DownAndInOptionGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "heston_3_2.down_and_in_option.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t heston_3_2_down_and_in_option_node_graph_workspace_bytes(
    const DownAndInOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::path_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        product::DownAndInOptionPathPolicy<Side>,
        DownAndInOptionGradientSchedule,
        11U,
        32U,
        2U
    >(host, configuration);
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_heston_3_2_down_and_in_option_node_graph_sensitivities_cuda(
    const DownAndInOptionPriceGradientPlan& host,
    DownAndInOptionPriceGradientPlan::DeviceInputs device,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_heston_3_2_down_and_in_option_cuda<Side>(
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
        product::DownAndInOptionPathPolicy<Side>,
        DownAndInOptionGradientSchedule,
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
        "heston_3_2.down_and_in_option.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}


template<OptionSide Side>
std::size_t heston_3_2_down_and_in_option_mixed_node_graph_workspace_bytes(
    const DownAndInOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::mixed_path_node_graph_workspace_bytes<
        mpg::CoupledDynamics,
        product::DownAndInOptionPathPolicy<Side>,
        DownAndInOptionGradientSchedule,
        11U,
        55U,
        128U,
        2U
    >(host, configuration);
}

template<OptionSide Side>
void launch_heston_3_2_down_and_in_option_mixed_node_graph_sensitivities_cuda(
    const DownAndInOptionPriceGradientPlan& host,
    DownAndInOptionPriceGradientPlan::DeviceInputs device,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    DownAndInOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    epg::launch_mixed_path_node_graph_sensitivities<
        mpg::CoupledDynamics,
        product::DownAndInOptionPathPolicy<Side>,
        DownAndInOptionGradientSchedule,
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
        "heston_3_2.down_and_in_option.sensitivities.mixed_node_graph",
        "full_hessian"
    );
}

template void launch_heston_3_2_down_and_in_option_price_gradients_cuda<OptionSide::call>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::StencilOutputs,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_heston_3_2_down_and_in_option_diagonal_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_3_2_down_and_in_option_node_graph_workspace_bytes<OptionSide::call, pg::SensitivityOrders::second>(
    const DownAndInOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_down_and_in_option_node_graph_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_heston_3_2_down_and_in_option_diagonal_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::first_and_second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_3_2_down_and_in_option_node_graph_workspace_bytes<OptionSide::call, pg::SensitivityOrders::first_and_second>(
    const DownAndInOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_down_and_in_option_node_graph_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::first_and_second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template std::size_t
heston_3_2_down_and_in_option_mixed_node_graph_workspace_bytes<OptionSide::call>(
    const DownAndInOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_down_and_in_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    DownAndInOptionPriceGradientPlan::MixedStencilOutputs,
    const pg::LaunchConfiguration&,
    pg::SensitivityOutputs, pg::MixedSensitivityOutputs,
    void*, std::size_t);

template void launch_heston_3_2_down_and_in_option_price_gradients_cuda<OptionSide::put>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::StencilOutputs,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_heston_3_2_down_and_in_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_3_2_down_and_in_option_node_graph_workspace_bytes<OptionSide::put, pg::SensitivityOrders::second>(
    const DownAndInOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_down_and_in_option_node_graph_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_heston_3_2_down_and_in_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_3_2_down_and_in_option_node_graph_workspace_bytes<OptionSide::put, pg::SensitivityOrders::first_and_second>(
    const DownAndInOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_down_and_in_option_node_graph_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template std::size_t
heston_3_2_down_and_in_option_mixed_node_graph_workspace_bytes<OptionSide::put>(
    const DownAndInOptionPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_down_and_in_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>(
    const DownAndInOptionPriceGradientPlan&,
    DownAndInOptionPriceGradientPlan::DeviceInputs,
    DownAndInOptionPriceGradientPlan::DiagonalStencilOutputs,
    DownAndInOptionPriceGradientPlan::MixedStencilOutputs,
    const pg::LaunchConfiguration&,
    pg::SensitivityOutputs, pg::MixedSensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::heston_3_2
