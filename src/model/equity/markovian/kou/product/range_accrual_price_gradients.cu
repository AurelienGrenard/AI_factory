// Generated kou range_accrual path sensitivities over the common device-prepared engine.
#include "model/equity/markovian/kou/product/range_accrual_price_gradients.cuh"

#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/path_node_graph_launcher.cuh"
#include "common/equity/price_gradients/mixed_path_node_graph_launcher.cuh"
#include "model/equity/markovian/kou/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/kou/product/range_accrual.cuh"
#include "product/range_accrual/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::kou {

using RangeAccrualGradientSchedule = simulation::ExactTransitionRegularSchedule<kou::DynamicsPolicy>;

void prepare_range_accrual_price_gradient_stencils_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "kou.range_accrual.price_gradients.stencil_preparation"
    );
}

void prepare_range_accrual_diagonal_sensitivity_stencils_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "kou.range_accrual.diagonal.stencil_preparation"
    );
}


void launch_kou_range_accrual_price_gradients_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_kou_range_accrual_cuda(
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
    epg::launch_path_first_sensitivities<
        mpg::CoupledDynamics,
        product::RangeAccrualPathPolicy,
        RangeAccrualGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "kou.range_accrual.price_gradients.device_prepared",
        "default/nodes=3/B=1"
    );
}

template<pg::SensitivityOrders Orders>
void launch_kou_range_accrual_diagonal_sensitivities_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_kou_range_accrual_cuda(
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
    epg::launch_path_diagonal_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        product::RangeAccrualPathPolicy,
        RangeAccrualGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "kou.range_accrual.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<pg::SensitivityOrders Orders>
std::size_t kou_range_accrual_node_graph_workspace_bytes(
    const RangeAccrualPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::path_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        product::RangeAccrualPathPolicy,
        RangeAccrualGradientSchedule,
        11U,
        32U,
        2U
    >(host, configuration);
}

template<pg::SensitivityOrders Orders>
void launch_kou_range_accrual_node_graph_sensitivities_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_kou_range_accrual_cuda(
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
    epg::launch_path_node_graph_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        product::RangeAccrualPathPolicy,
        RangeAccrualGradientSchedule,
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
        "kou.range_accrual.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}



std::size_t kou_range_accrual_mixed_node_graph_workspace_bytes(
    const RangeAccrualPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::mixed_path_node_graph_workspace_bytes<
        mpg::CoupledDynamics,
        product::RangeAccrualPathPolicy,
        RangeAccrualGradientSchedule,
        11U,
        55U,
        128U,
        2U
    >(host, configuration);
}


void launch_kou_range_accrual_mixed_node_graph_sensitivities_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    RangeAccrualPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    epg::launch_mixed_path_node_graph_sensitivities<
        mpg::CoupledDynamics,
        product::RangeAccrualPathPolicy,
        RangeAccrualGradientSchedule,
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
        "kou.range_accrual.sensitivities.mixed_node_graph",
        "full_hessian"
    );
}

template void launch_kou_range_accrual_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const RangeAccrualPriceGradientPlan&,
    RangeAccrualPriceGradientPlan::DeviceInputs,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
kou_range_accrual_node_graph_workspace_bytes<pg::SensitivityOrders::second>(
    const RangeAccrualPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_kou_range_accrual_node_graph_sensitivities_cuda<pg::SensitivityOrders::second>(
    const RangeAccrualPriceGradientPlan&,
    RangeAccrualPriceGradientPlan::DeviceInputs,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_kou_range_accrual_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const RangeAccrualPriceGradientPlan&,
    RangeAccrualPriceGradientPlan::DeviceInputs,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
kou_range_accrual_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>(
    const RangeAccrualPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_kou_range_accrual_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const RangeAccrualPriceGradientPlan&,
    RangeAccrualPriceGradientPlan::DeviceInputs,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::kou
