// Generated Black-Scholes European composition over analytical and MC gradient engines.
#include "model/equity/markovian/${model}/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/${model}/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/${model}/analytics_impl.cuh"
#include "common/equity/price_gradients/terminal_policy.cuh"
#include "common/monte_carlo/price_gradients/kernel.cuh"
#include "common/closed_form/price_gradients/kernel.cuh"
#include "product/european_option/price_gradients/monte_carlo_policy.cuh"
#include "product/european_option/price_gradients/closed_form_policy.cuh"

namespace ai_factory::workbench::model::equity::${model} {
namespace pg = ::ai_factory::workbench::price_gradients;
template<OptionSide Side>
void launch_${model}_european_option_price_gradients_cuda(
    const EuropeanOptionPriceGradientPlan& host, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType> device,
    const pg::LaunchConfiguration& configuration, pg::Outputs outputs
) {
    pg::validate_launch(host, device, configuration, outputs);
    if (configuration.method == pg::PricingMethod::closed_form) {
        pg::dispatch_count<${maximum}>(host.sensitivity_count(), [&](auto cardinality) {
            using Policy = product::european_option::price_gradients::ClosedFormPolicy<ModelParameters, Side>;
            closed_form::price_gradients::launch<Policy, decltype(cardinality)::value>(device, configuration, outputs,
                "${model}.european_option.price_gradients.closed_form", option_side_name(Side));
        });
        return;
    }
    monte_carlo::price_gradients::launch<price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>>(device, configuration, host.time, outputs,
            host.sensitivity_count(), "${model}.european_option.price_gradients.monte_carlo", option_side_name(Side));
}
template void launch_${model}_european_option_price_gradients_cuda<OptionSide::call>(
    const EuropeanOptionPriceGradientPlan&, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType>,
    const pg::LaunchConfiguration&, pg::Outputs);
template void launch_${model}_european_option_price_gradients_cuda<OptionSide::put>(
    const EuropeanOptionPriceGradientPlan&, pg::DeviceInputs<EuropeanOptionPriceGradientPlan::ScenarioType>,
    const pg::LaunchConfiguration&, pg::Outputs);
}  // namespace ai_factory::workbench::model::equity::${model}
