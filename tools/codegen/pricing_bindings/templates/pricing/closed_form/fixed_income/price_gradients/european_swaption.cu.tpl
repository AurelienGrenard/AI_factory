// ${model_display}${curve_display_suffix} European-swaption analytical sensitivities.
#include "${unit_path}.cuh"

#include "common/closed_form/price_gradients/device_prepared_cooperative_kernel.cuh"
#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/fixed_income/price_gradients/closed_form_policy.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "product/european_swaption/pricing_policy.cuh"
${implementation_include}
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::${binding_namespace} {

void prepare_european_swaption_price_gradient_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host, device, stencil_outputs, result_offset, result_count,
        "${diagnostic_name}.sensitivities.prepare_stencils"
    );
}

void prepare_european_swaption_diagonal_sensitivity_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "${diagnostic_name}.sensitivities.prepare_diagonal_stencils"
    );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders, typename Outputs>
void launch_sensitivities(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    ::ai_factory::workbench::monte_carlo::price_gradients::
        DevicePreparedStencilOutputs<
            pg::SensitivityTraits<Orders>::node_capacity
        > stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    Outputs outputs,
    closed_form::WorkDistribution distribution
) {
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "${diagnostic_name} Jamshidian sensitivities require closed-form pricing."
        );
    }
    if (distribution != closed_form::WorkDistribution::scalar
        && distribution != closed_form::WorkDistribution::cooperative) {
        throw std::invalid_argument(
            "Unknown ${diagnostic_name} sensitivity work distribution."
        );
    }
    using PricingPolicy = ${pricing_policy};
    using Policy = ${scenario_policy};
    const auto sensitivity_outputs = pg::as_sensitivity_outputs(outputs);
    const bool cooperative =
        distribution == closed_form::WorkDistribution::cooperative
        && host.maximum_payment_count > 1U
        && closed_form::price_gradients::
            launch_device_prepared_cooperative<Orders, Policy>(
                host,
                device,
                stencil_outputs,
                configuration,
                sensitivity_outputs,
                host.maximum_payment_count,
                "${diagnostic_name}.sensitivities.cooperative",
                swaption_side_name(Side)
            );
    if (cooperative) return;
    closed_form::price_gradients::launch_device_prepared<Orders, Policy>(
        host,
        device,
        stencil_outputs,
        configuration,
        sensitivity_outputs,
        "${diagnostic_name}.sensitivities.scalar",
        swaption_side_name(Side)
    );
}

template<SwaptionSide Side>
void launch_${function_prefix}_european_swaption_price_gradients_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs,
    closed_form::WorkDistribution distribution
) {
    launch_sensitivities<Side, pg::SensitivityOrders::first>(
        host, device, stencil_outputs, configuration, outputs, distribution
    );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
void launch_${function_prefix}_european_swaption_diagonal_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    closed_form::WorkDistribution distribution
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_sensitivities<Side, Orders>(
        host, device, stencil_outputs, configuration, outputs, distribution
    );
}

#define AI_FACTORY_INSTANTIATE(side)                                        \
template void launch_${function_prefix}_european_swaption_price_gradients_cuda< \
    side>(const EuropeanSwaptionPriceGradientPlan&,                         \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::StencilOutputs,                      \
    const pg::LaunchConfiguration&, pg::Outputs,                            \
    closed_form::WorkDistribution);                                         \
template void launch_${function_prefix}_european_swaption_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::second>(                                   \
    const EuropeanSwaptionPriceGradientPlan&,                               \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs,              \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,                 \
    closed_form::WorkDistribution);                                         \
template void launch_${function_prefix}_european_swaption_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::first_and_second>(                         \
    const EuropeanSwaptionPriceGradientPlan&,                               \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs,              \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,                 \
    closed_form::WorkDistribution)

AI_FACTORY_INSTANTIATE(SwaptionSide::payer);
AI_FACTORY_INSTANTIATE(SwaptionSide::receiver);
#undef AI_FACTORY_INSTANTIATE

}  // namespace ai_factory::workbench::model::fixed_income::${binding_namespace}
