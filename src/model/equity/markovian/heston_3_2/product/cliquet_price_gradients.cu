// Generated heston_3_2 cliquet path sensitivities over the common device-prepared engine.
#include "model/equity/markovian/heston_3_2/product/cliquet_price_gradients.cuh"

#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/path_node_graph_launcher.cuh"
#include "common/equity/price_gradients/mixed_path_node_graph_launcher.cuh"
#include "model/equity/markovian/heston_3_2/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/heston_3_2/product/cliquet.cuh"
#include "product/cliquet/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::heston_3_2 {

using CliquetGradientSchedule = simulation::FixedStepRegularSchedule<heston_3_2::DynamicsPolicy>;

void prepare_cliquet_price_gradient_stencils_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "heston_3_2.cliquet.price_gradients.stencil_preparation"
    );
}

void prepare_cliquet_diagonal_sensitivity_stencils_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "heston_3_2.cliquet.diagonal.stencil_preparation"
    );
}


void launch_heston_3_2_cliquet_price_gradients_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_heston_3_2_cliquet_cuda(
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
        product::CliquetPathPolicy,
        CliquetGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "heston_3_2.cliquet.price_gradients.device_prepared",
        "default/nodes=3/B=1"
    );
}

template<pg::SensitivityOrders Orders>
void launch_heston_3_2_cliquet_diagonal_sensitivities_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_heston_3_2_cliquet_cuda(
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
        product::CliquetPathPolicy,
        CliquetGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "heston_3_2.cliquet.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<pg::SensitivityOrders Orders>
std::size_t heston_3_2_cliquet_node_graph_workspace_bytes(
    const CliquetPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::path_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        product::CliquetPathPolicy,
        CliquetGradientSchedule,
        14U,
        32U,
        2U
    >(host, configuration);
}

template<pg::SensitivityOrders Orders>
void launch_heston_3_2_cliquet_node_graph_sensitivities_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_heston_3_2_cliquet_cuda(
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
        product::CliquetPathPolicy,
        CliquetGradientSchedule,
        14U,
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
        "heston_3_2.cliquet.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}



std::size_t heston_3_2_cliquet_mixed_node_graph_workspace_bytes(
    const CliquetPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::mixed_path_node_graph_workspace_bytes<
        mpg::CoupledDynamics,
        product::CliquetPathPolicy,
        CliquetGradientSchedule,
        14U,
        91U,
        128U,
        4U
    >(host, configuration);
}


void launch_heston_3_2_cliquet_mixed_node_graph_sensitivities_cuda(
    const CliquetPriceGradientPlan& host,
    CliquetPriceGradientPlan::DeviceInputs device,
    CliquetPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    CliquetPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    epg::launch_mixed_path_node_graph_sensitivities<
        mpg::CoupledDynamics,
        product::CliquetPathPolicy,
        CliquetGradientSchedule,
        14U,
        91U,
        128U,
        4U
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
        "heston_3_2.cliquet.sensitivities.mixed_node_graph",
        "full_hessian"
    );
}

template void launch_heston_3_2_cliquet_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const CliquetPriceGradientPlan&,
    CliquetPriceGradientPlan::DeviceInputs,
    CliquetPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_3_2_cliquet_node_graph_workspace_bytes<pg::SensitivityOrders::second>(
    const CliquetPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_cliquet_node_graph_sensitivities_cuda<pg::SensitivityOrders::second>(
    const CliquetPriceGradientPlan&,
    CliquetPriceGradientPlan::DeviceInputs,
    CliquetPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_heston_3_2_cliquet_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const CliquetPriceGradientPlan&,
    CliquetPriceGradientPlan::DeviceInputs,
    CliquetPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
heston_3_2_cliquet_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>(
    const CliquetPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_heston_3_2_cliquet_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const CliquetPriceGradientPlan&,
    CliquetPriceGradientPlan::DeviceInputs,
    CliquetPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::heston_3_2
