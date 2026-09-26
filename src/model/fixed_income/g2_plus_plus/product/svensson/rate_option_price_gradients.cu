// G2++/Svensson rate_option analytical sensitivities.
#include "model/fixed_income/g2_plus_plus/product/svensson/rate_option_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/fixed_income/price_gradients/closed_form_policy.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "product/rate_option/pricing_policy.cuh"
#include "model/fixed_income/g2_plus_plus/svensson/analytics_impl.cuh"
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson {

void prepare_rate_option_price_gradient_stencils_cuda(
    const RateOptionPriceGradientPlan& host,
    RateOptionPriceGradientPlan::DeviceInputs device,
    RateOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host, device, stencil_outputs, result_offset, result_count,
        "g2_plus_plus.svensson.rate_option.sensitivities.prepare_stencils"
    );
}

void prepare_rate_option_diagonal_sensitivity_stencils_cuda(
    const RateOptionPriceGradientPlan& host,
    RateOptionPriceGradientPlan::DeviceInputs device,
    RateOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "g2_plus_plus.svensson.rate_option.sensitivities.prepare_diagonal_stencils"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders, typename Outputs>
void launch_sensitivities(
    const RateOptionPriceGradientPlan& host,
    RateOptionPriceGradientPlan::DeviceInputs device,
    ::ai_factory::workbench::monte_carlo::price_gradients::
        DevicePreparedStencilOutputs<
            pg::SensitivityTraits<Orders>::node_capacity
        > stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    Outputs outputs
) {
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "g2_plus_plus.svensson.rate_option sensitivities require closed-form pricing."
        );
    }
    using PricingPolicy = ::ai_factory::workbench::fixed_income::FittedRateOptionClosedFormPricingPolicy<
        FittedModelComposition, Side
    >;
    using Policy = ::ai_factory::workbench::fixed_income::price_gradients::
    CurveScalarScenarioClosedFormPolicy<
        PricingPolicy, ModelParameters, CurveParameters,
        product::RateOptionParameters
    >;
    closed_form::price_gradients::launch_device_prepared<Orders, Policy>(
        host,
        device,
        stencil_outputs,
        configuration,
        pg::as_sensitivity_outputs(outputs),
        "g2_plus_plus.svensson.rate_option.sensitivities.closed_form",
        option_side_name(Side)
    );
}

template<OptionSide Side>
void launch_g2_plus_plus_svensson_rate_option_price_gradients_cuda(
    const RateOptionPriceGradientPlan& host,
    RateOptionPriceGradientPlan::DeviceInputs device,
    RateOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    launch_sensitivities<Side, pg::SensitivityOrders::first>(
        host, device, stencil_outputs, configuration, outputs
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_g2_plus_plus_svensson_rate_option_diagonal_sensitivities_cuda(
    const RateOptionPriceGradientPlan& host,
    RateOptionPriceGradientPlan::DeviceInputs device,
    RateOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_sensitivities<Side, Orders>(
        host, device, stencil_outputs, configuration, outputs
    );
}

#define AI_FACTORY_INSTANTIATE(side)                                        \
template void launch_g2_plus_plus_svensson_rate_option_price_gradients_cuda<    \
    side>(const RateOptionPriceGradientPlan&,                           \
    RateOptionPriceGradientPlan::DeviceInputs,                          \
    RateOptionPriceGradientPlan::StencilOutputs,                        \
    const pg::LaunchConfiguration&, pg::Outputs);                            \
template void launch_g2_plus_plus_svensson_rate_option_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::second>(                                   \
    const RateOptionPriceGradientPlan&,                                 \
    RateOptionPriceGradientPlan::DeviceInputs,                          \
    RateOptionPriceGradientPlan::DiagonalStencilOutputs,                \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);                 \
template void launch_g2_plus_plus_svensson_rate_option_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::first_and_second>(                         \
    const RateOptionPriceGradientPlan&,                                 \
    RateOptionPriceGradientPlan::DeviceInputs,                          \
    RateOptionPriceGradientPlan::DiagonalStencilOutputs,                \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs)

AI_FACTORY_INSTANTIATE(OptionSide::call);
AI_FACTORY_INSTANTIATE(OptionSide::put);
#undef AI_FACTORY_INSTANTIATE

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson
