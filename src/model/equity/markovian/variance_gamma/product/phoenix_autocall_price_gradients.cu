// Generated variance_gamma phoenix_autocall path sensitivities over the common device-prepared engine.
#include "model/equity/markovian/variance_gamma/product/phoenix_autocall_price_gradients.cuh"

#include "common/equity/price_gradients/path_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/path_node_graph_launcher.cuh"
#include "model/equity/markovian/variance_gamma/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/variance_gamma/product/phoenix_autocall.cuh"
#include "product/phoenix_autocall/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::variance_gamma {

using PhoenixAutocallGradientSchedule = simulation::ExactTransitionRegularSchedule<variance_gamma::DynamicsPolicy>;

void prepare_phoenix_autocall_price_gradient_stencils_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "variance_gamma.phoenix_autocall.price_gradients.stencil_preparation"
    );
}

void prepare_phoenix_autocall_diagonal_sensitivity_stencils_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "variance_gamma.phoenix_autocall.diagonal.stencil_preparation"
    );
}


void launch_variance_gamma_phoenix_autocall_price_gradients_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_variance_gamma_phoenix_autocall_cuda(
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
        product::PhoenixAutocallPathPolicy,
        PhoenixAutocallGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "variance_gamma.phoenix_autocall.price_gradients.device_prepared",
        "default/nodes=3/B=1"
    );
}

template<pg::SensitivityOrders Orders>
void launch_variance_gamma_phoenix_autocall_diagonal_sensitivities_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_variance_gamma_phoenix_autocall_cuda(
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
        product::PhoenixAutocallPathPolicy,
        PhoenixAutocallGradientSchedule
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "variance_gamma.phoenix_autocall.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<pg::SensitivityOrders Orders>
std::size_t variance_gamma_phoenix_autocall_node_graph_workspace_bytes(
    const PhoenixAutocallPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return epg::path_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        product::PhoenixAutocallPathPolicy,
        PhoenixAutocallGradientSchedule,
        10U,
        16U,
        2U
    >(host, configuration);
}

template<pg::SensitivityOrders Orders>
void launch_variance_gamma_phoenix_autocall_node_graph_sensitivities_cuda(
    const PhoenixAutocallPriceGradientPlan& host,
    PhoenixAutocallPriceGradientPlan::DeviceInputs device,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    const auto launch_price_only = [&] {
        launch_variance_gamma_phoenix_autocall_cuda(
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
        product::PhoenixAutocallPathPolicy,
        PhoenixAutocallGradientSchedule,
        10U,
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
        "variance_gamma.phoenix_autocall.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}

template void launch_variance_gamma_phoenix_autocall_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const PhoenixAutocallPriceGradientPlan&,
    PhoenixAutocallPriceGradientPlan::DeviceInputs,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
variance_gamma_phoenix_autocall_node_graph_workspace_bytes<pg::SensitivityOrders::second>(
    const PhoenixAutocallPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_variance_gamma_phoenix_autocall_node_graph_sensitivities_cuda<pg::SensitivityOrders::second>(
    const PhoenixAutocallPriceGradientPlan&,
    PhoenixAutocallPriceGradientPlan::DeviceInputs,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);
template void launch_variance_gamma_phoenix_autocall_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const PhoenixAutocallPriceGradientPlan&,
    PhoenixAutocallPriceGradientPlan::DeviceInputs,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template std::size_t
variance_gamma_phoenix_autocall_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>(
    const PhoenixAutocallPriceGradientPlan&,
    const pg::LaunchConfiguration&);
template void launch_variance_gamma_phoenix_autocall_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const PhoenixAutocallPriceGradientPlan&,
    PhoenixAutocallPriceGradientPlan::DeviceInputs,
    PhoenixAutocallPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,
    void*, std::size_t);

}  // namespace ai_factory::workbench::model::equity::variance_gamma
