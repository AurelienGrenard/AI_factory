// cir European-swaption sensitivities over scalar/cooperative Jamshidian.
#include "model/fixed_income/cir/product/european_swaption_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_cooperative_kernel.cuh"
#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/fixed_income/price_gradients/closed_form_policy.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "product/european_swaption/pricing_policy.cuh"

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::cir {
namespace pg = ::ai_factory::workbench::price_gradients;

void prepare_european_swaption_price_gradient_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "cir.european_swaption.sensitivities.prepare_stencils"
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
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "cir.european_swaption.sensitivities.prepare_diagonal_stencils"
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
            "cir Jamshidian sensitivities require closed-form pricing."
        );
    }
    if (distribution != closed_form::WorkDistribution::scalar
        && distribution != closed_form::WorkDistribution::cooperative) {
        throw std::invalid_argument(
            "Unknown cir sensitivity work distribution."
        );
    }

    using SidePricingPolicy = ::ai_factory::workbench::fixed_income::
        CooperativeOneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side,
            AnalyticsProvider,
            ModelParameters,
            product::RegularEuropeanSwaptionParameters,
            product::RegularEuropeanSwaptionScheduleSource
        >;
    using Policy = ::ai_factory::workbench::fixed_income::price_gradients::
        ScenarioClosedFormPolicy<
            SidePricingPolicy,
            ModelParameters,
            product::RegularEuropeanSwaptionParameters,
            product::RegularEuropeanSwaptionScheduleSource
        >;
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
                "cir.european_swaption.sensitivities.cooperative",
                swaption_side_name(Side)
            );
    if (cooperative) return;
    closed_form::price_gradients::launch_device_prepared<Orders, Policy>(
        host,
        device,
        stencil_outputs,
        configuration,
        sensitivity_outputs,
        "cir.european_swaption.sensitivities.scalar",
        swaption_side_name(Side)
    );
}

template<SwaptionSide Side>
void launch_cir_european_swaption_price_gradients_cuda(
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
void launch_cir_european_swaption_diagonal_sensitivities_cuda(
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

#define AI_FACTORY_INSTANTIATE_FIRST(side)                                  \
template void launch_cir_european_swaption_price_gradients_cuda<      \
    side                                                                    \
>(                                                                          \
    const EuropeanSwaptionPriceGradientPlan&,                               \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::StencilOutputs,                      \
    const pg::LaunchConfiguration&,                                         \
    pg::Outputs,                                                            \
    closed_form::WorkDistribution                                           \
)

#define AI_FACTORY_INSTANTIATE_DIAGONAL(side, orders)                       \
template void launch_cir_european_swaption_diagonal_sensitivities_cuda<\
    side, orders                                                            \
>(                                                                          \
    const EuropeanSwaptionPriceGradientPlan&,                               \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs,              \
    const pg::LaunchConfiguration&,                                         \
    pg::SensitivityOutputs,                                                 \
    closed_form::WorkDistribution                                           \
)

AI_FACTORY_INSTANTIATE_FIRST(SwaptionSide::payer);
AI_FACTORY_INSTANTIATE_FIRST(SwaptionSide::receiver);
AI_FACTORY_INSTANTIATE_DIAGONAL(
    SwaptionSide::payer, pg::SensitivityOrders::second
);
AI_FACTORY_INSTANTIATE_DIAGONAL(
    SwaptionSide::receiver, pg::SensitivityOrders::second
);
AI_FACTORY_INSTANTIATE_DIAGONAL(
    SwaptionSide::payer, pg::SensitivityOrders::first_and_second
);
AI_FACTORY_INSTANTIATE_DIAGONAL(
    SwaptionSide::receiver, pg::SensitivityOrders::first_and_second
);

#undef AI_FACTORY_INSTANTIATE_DIAGONAL
#undef AI_FACTORY_INSTANTIATE_FIRST

}  // namespace ai_factory::workbench::model::fixed_income::cir
