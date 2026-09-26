// Generated Black-Scholes closed-form geometric_asian_option sensitivities.
#include "model/equity/markovian/black_scholes/product/geometric_asian_option_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/equity/price_gradients/scenario_closed_form_policy.cuh"
#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "model/equity/markovian/black_scholes/product/geometric_asian_option_impl.cuh"

#include <stdexcept>

namespace ai_factory::workbench::model::equity::black_scholes {

void prepare_geometric_asian_option_price_gradient_stencils_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "black_scholes.geometric_asian_option.price_gradients.stencil_preparation"
    );
}

void prepare_geometric_asian_option_diagonal_sensitivity_stencils_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "black_scholes.geometric_asian_option.diagonal.stencil_preparation"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders, typename StencilOutputs>
void launch_geometric_asian_option_closed_form_sensitivities(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    using ScenarioClosedFormPolicy =
        epg::ScenarioClosedFormPolicy<GeometricAsianOptionClosedFormPricingPolicy<Side>>;
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "black_scholes geometric_asian_option sensitivities require closed-form pricing."
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
        "black_scholes.geometric_asian_option.sensitivities.closed_form",
        Orders == pg::SensitivityOrders::first
            ? "gradient/nodes=3"
            : Orders == pg::SensitivityOrders::second
                ? "diagonal_hessian/nodes=4"
                : "gradient_and_diagonal_hessian/nodes=4"
    );
}

template<OptionSide Side>
void launch_black_scholes_geometric_asian_option_price_gradients_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    launch_geometric_asian_option_closed_form_sensitivities<
        Side, pg::SensitivityOrders::first
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        pg::as_sensitivity_outputs(outputs)
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_black_scholes_geometric_asian_option_diagonal_sensitivities_cuda(
    const GeometricAsianOptionPriceGradientPlan& host,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs device,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_geometric_asian_option_closed_form_sensitivities<
        Side, Orders
    >(
        host, device, stencil_outputs, configuration, outputs
    );
}

template void launch_black_scholes_geometric_asian_option_price_gradients_cuda<OptionSide::call>(
    const GeometricAsianOptionPriceGradientPlan&,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs,
    GeometricAsianOptionPriceGradientPlan::StencilOutputs,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_black_scholes_geometric_asian_option_diagonal_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::second>(
    const GeometricAsianOptionPriceGradientPlan&,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template void launch_black_scholes_geometric_asian_option_diagonal_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::first_and_second>(
    const GeometricAsianOptionPriceGradientPlan&,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);

template void launch_black_scholes_geometric_asian_option_price_gradients_cuda<OptionSide::put>(
    const GeometricAsianOptionPriceGradientPlan&,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs,
    GeometricAsianOptionPriceGradientPlan::StencilOutputs,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_black_scholes_geometric_asian_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::second>(
    const GeometricAsianOptionPriceGradientPlan&,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);
template void launch_black_scholes_geometric_asian_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>(
    const GeometricAsianOptionPriceGradientPlan&,
    GeometricAsianOptionPriceGradientPlan::DeviceInputs,
    GeometricAsianOptionPriceGradientPlan::DiagonalStencilOutputs,
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);

}  // namespace ai_factory::workbench::model::equity::black_scholes
