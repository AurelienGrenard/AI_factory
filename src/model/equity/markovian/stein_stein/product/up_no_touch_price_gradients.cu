// Generated stein_stein up_no_touch path sensitivities over the common device-prepared engine.
#include "model/equity/markovian/stein_stein/product/up_no_touch_price_gradients.cuh"

#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/path_node_graph_launcher.cuh"
#include "model/equity/markovian/stein_stein/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/stein_stein/product/up_no_touch.cuh"
#include "product/up_no_touch/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::stein_stein {

using UpNoTouchGradientSchedule = simulation::FixedStepDenseSchedule<stein_stein::DynamicsPolicy>;

void prepare_up_no_touch_price_gradient_stencils_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "stein_stein.up_no_touch.price_gradients.stencil_preparation"
    );
}

void prepare_up_no_touch_diagonal_sensitivity_stencils_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "stein_stein.up_no_touch.diagonal.stencil_preparation"
    );
}


void launch_stein_stein_up_no_touch_price_gradients_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_stein_stein_up_no_touch_cuda(
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
        product::UpNoTouchPathPolicy,
        UpNoTouchGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "stein_stein.up_no_touch.price_gradients.device_prepared",
        "default/nodes=3/B=1"
    );
}

template<pg::SensitivityOrders Orders>
void launch_stein_stein_up_no_touch_diagonal_sensitivities_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_stein_stein_up_no_touch_cuda(
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
        product::UpNoTouchPathPolicy,
        UpNoTouchGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "stein_stein.up_no_touch.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<pg::SensitivityOrders Orders>
std::size_t stein_stein_up_no_touch_node_graph_workspace_bytes(
    const UpNoTouchPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::path_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        product::UpNoTouchPathPolicy,
        UpNoTouchGradientSchedule,
        9U,
        16U,
        2U
    >(host, configuration);
}

template<pg::SensitivityOrders Orders>
void launch_stein_stein_up_no_touch_node_graph_sensitivities_cuda(
    const UpNoTouchPriceGradientPlan& host,
    UpNoTouchPriceGradientPlan::DeviceInputs device,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_stein_stein_up_no_touch_cuda(
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
        product::UpNoTouchPathPolicy,
        UpNoTouchGradientSchedule,
        9U,
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
        "stein_stein.up_no_touch.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}

template void launch_stein_stein_up_no_touch_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const UpNoTouchPriceGradientPlan&,
    UpNoTouchPriceGradientPlan::DeviceInputs,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
stein_stein_up_no_touch_node_graph_workspace_bytes<pg::SensitivityOrders::second>(
    const UpNoTouchPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_stein_stein_up_no_touch_node_graph_sensitivities_cuda<pg::SensitivityOrders::second>(
    const UpNoTouchPriceGradientPlan&,
    UpNoTouchPriceGradientPlan::DeviceInputs,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_stein_stein_up_no_touch_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const UpNoTouchPriceGradientPlan&,
    UpNoTouchPriceGradientPlan::DeviceInputs,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
stein_stein_up_no_touch_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>(
    const UpNoTouchPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_stein_stein_up_no_touch_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const UpNoTouchPriceGradientPlan&,
    UpNoTouchPriceGradientPlan::DeviceInputs,
    UpNoTouchPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::stein_stein
