// Generated Black-Scholes European composition over analytical and MC gradient engines.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"
#include "common/equity/price_gradients/terminal_policy.cuh"
#include "common/monte_carlo/price_gradients/kernel.cuh"
#include "common/closed_form/price_gradients/kernel.cuh"
#include "product/european_option/price_gradients/monte_carlo_policy.cuh"
#include "product/european_option/price_gradients/closed_form_policy.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace pg = ::ai_factory::workbench::price_gradients;
template<OptionSide Side>
void launch_black_scholes_european_option_price_gradients_cuda(
    const EuropeanOptionPriceGradientPlan& host, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType> device,
    const pg::LaunchConfiguration& configuration, pg::Outputs outputs
) {
    pg::validate_launch(host, device, configuration, outputs);
    if (configuration.method == pg::PricingMethod::closed_form) {
        pg::dispatch_count<6>(host.sensitivity_count(), [&](auto cardinality) {
            using Policy = product::european_option::price_gradients::ClosedFormPolicy<ModelParameters, Side>;
            closed_form::price_gradients::launch<Policy, decltype(cardinality)::value>(device, configuration, outputs,
                "black_scholes.european_option.price_gradients.closed_form", option_side_name(Side));
        });
        return;
    }
    monte_carlo::price_gradients::launch<price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>>(device, configuration, host.time, outputs,
            host.sensitivity_count(), "black_scholes.european_option.price_gradients.monte_carlo", option_side_name(Side));
}
template void launch_black_scholes_european_option_price_gradients_cuda<OptionSide::call>(
    const EuropeanOptionPriceGradientPlan&, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType>,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_black_scholes_european_option_price_gradients_cuda<OptionSide::put>(
    const EuropeanOptionPriceGradientPlan&, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType>,
    const pg::LaunchConfiguration&, pg::Outputs);
}  // namespace ai_factory::workbench::model::equity::black_scholes
