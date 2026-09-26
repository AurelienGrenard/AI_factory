// Generated Black-Scholes closed-form range_accrual sensitivities.
#include "model/equity/markovian/black_scholes/product/range_accrual_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/closed_form/price_gradients/device_prepared_mixed_kernel.cuh"
#include "common/equity/price_gradients/scenario_closed_form_policy.cuh"
#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "model/equity/markovian/black_scholes/product/range_accrual_impl.cuh"

#include <stdexcept>

namespace ai_factory::workbench::model::equity::black_scholes {

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
        "black_scholes.range_accrual.price_gradients.stencil_preparation"
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
        "black_scholes.range_accrual.diagonal.stencil_preparation"
    );
}

template<pg::SensitivityOrders Orders, typename StencilOutputs>
void launch_range_accrual_closed_form_sensitivities(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    using ScenarioClosedFormPolicy =
        epg::ScenarioClosedFormPolicy<RangeAccrualClosedFormPricingPolicy>;
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "black_scholes range_accrual sensitivities require closed-form pricing."
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
        "black_scholes.range_accrual.sensitivities.closed_form",
        Orders == pg::SensitivityOrders::first
            ? "gradient/nodes=3"
            : Orders == pg::SensitivityOrders::second
                ? "diagonal_hessian/nodes=4"
                : "gradient_and_diagonal_hessian/nodes=4"
    );
}


void launch_black_scholes_range_accrual_price_gradients_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    launch_range_accrual_closed_form_sensitivities<
        pg::SensitivityOrders::first
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        pg::as_sensitivity_outputs(outputs)
    );
}

template<pg::SensitivityOrders Orders>
void launch_black_scholes_range_accrual_diagonal_sensitivities_cuda(
    const RangeAccrualPriceGradientPlan& host,
    RangeAccrualPriceGradientPlan::DeviceInputs device,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_range_accrual_closed_form_sensitivities<
        Orders
    >(
        host, device, stencil_outputs, configuration, outputs
    );
}


std::size_t black_scholes_range_accrual_mixed_node_graph_workspace_bytes(
    const RangeAccrualPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    (void)configuration;
    return closed_form::price_gradients::mixed_workspace_bytes(host);
}


void launch_black_scholes_range_accrual_mixed_node_graph_sensitivities_cuda(
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
    using ScenarioClosedFormPolicy =
        epg::ScenarioClosedFormPolicy<RangeAccrualClosedFormPricingPolicy>;
    closed_form::price_gradients::launch_device_prepared_mixed<
        ScenarioClosedFormPolicy,
        7U,
        21U
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
        "black_scholes.range_accrual.sensitivities.closed_form_mixed",
        "selected_gradient_and_hessian"
    );
}

template void launch_black_scholes_range_accrual_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const RangeAccrualPriceGradientPlan&,
    RangeAccrualPriceGradientPlan::DeviceInputs,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template void launch_black_scholes_range_accrual_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const RangeAccrualPriceGradientPlan&,
    RangeAccrualPriceGradientPlan::DeviceInputs,
    RangeAccrualPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);

}  // namespace ai_factory::workbench::model::equity::black_scholes
