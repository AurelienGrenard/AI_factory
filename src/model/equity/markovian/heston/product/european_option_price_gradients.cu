// Generated heston European composition over the common selected-gradient MC engine.
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/price_gradients/coupled_dynamics_impl.cuh"
#include "common/equity/price_gradients/terminal_policy.cuh"
#include "common/monte_carlo/price_gradients/kernel.cuh"
#include "product/european_option/price_gradients/monte_carlo_policy.cuh"

namespace ai_factory::workbench::model::equity::heston {
namespace pg = ::ai_factory::workbench::price_gradients;
template<OptionSide Side>
void launch_heston_european_option_price_gradients_cuda(
    const EuropeanOptionPriceGradientPlan& host, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType> device,
    const pg::LaunchConfiguration& configuration, pg::Outputs outputs
) {
    pg::validate_launch(host, device, configuration, outputs);
    if (configuration.method != pg::PricingMethod::monte_carlo)
        throw std::invalid_argument("heston European gradients require Monte Carlo.");
    monte_carlo::price_gradients::launch<price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>>(device, configuration, host.time, outputs,
            host.sensitivity_count(), "heston.european_option.price_gradients.monte_carlo", option_side_name(Side));
}
template void launch_heston_european_option_price_gradients_cuda<OptionSide::call>(
    const EuropeanOptionPriceGradientPlan&, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType>,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_heston_european_option_price_gradients_cuda<OptionSide::put>(
    const EuropeanOptionPriceGradientPlan&, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType>,
    const pg::LaunchConfiguration&, pg::Outputs);
}  // namespace ai_factory::workbench::model::equity::heston
