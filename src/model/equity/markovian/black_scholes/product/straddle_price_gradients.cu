// Generated Black-Scholes closed-form straddle sensitivities.
#include "model/equity/markovian/black_scholes/product/straddle_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/equity/price_gradients/scenario_closed_form_policy.cuh"
#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "model/equity/markovian/black_scholes/product/straddle_impl.cuh"

#include <stdexcept>

namespace ai_factory::workbench::model::equity::black_scholes {

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
        "black_scholes.straddle.price_gradients.stencil_preparation"
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
        "black_scholes.straddle.diagonal.stencil_preparation"
    );
}

template<pg::SensitivityOrders Orders, typename StencilOutputs>
void launch_straddle_closed_form_sensitivities(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    using ScenarioClosedFormPolicy =
        epg::ScenarioClosedFormPolicy<StraddleClosedFormPricingPolicy>;
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "black_scholes straddle sensitivities require closed-form pricing."
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
        "black_scholes.straddle.sensitivities.closed_form",
        Orders == pg::SensitivityOrders::first
            ? "gradient/nodes=3"
            : Orders == pg::SensitivityOrders::second
                ? "diagonal_hessian/nodes=4"
                : "gradient_and_diagonal_hessian/nodes=4"
    );
}


void launch_black_scholes_straddle_price_gradients_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    launch_straddle_closed_form_sensitivities<
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
void launch_black_scholes_straddle_diagonal_sensitivities_cuda(
    const StraddlePriceGradientPlan& host,
    StraddlePriceGradientPlan::DeviceInputs device,
    StraddlePriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_straddle_closed_form_sensitivities<
        Orders
    >(
        host, device, stencil_outputs, configuration, outputs
    );
}

template void launch_black_scholes_straddle_diagonal_sensitivities_cuda<pg::SensitivityOrders::second>(
    const StraddlePriceGradientPlan&,
    StraddlePriceGradientPlan::DeviceInputs,
    StraddlePriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template void launch_black_scholes_straddle_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>(
    const StraddlePriceGradientPlan&,
    StraddlePriceGradientPlan::DeviceInputs,
    StraddlePriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);

}  // namespace ai_factory::workbench::model::equity::black_scholes
