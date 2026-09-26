// OU zero_coupon_bond_option analytical sensitivities.
#include "model/fixed_income/ornstein_uhlenbeck/product/zero_coupon_bond_option_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/fixed_income/price_gradients/closed_form_policy.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "product/zero_coupon_bond_option/pricing_policy.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/analytics_impl.cuh"
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {

void prepare_zero_coupon_bond_option_price_gradient_stencils_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host, device, stencil_outputs, result_offset, result_count,
        "ornstein_uhlenbeck.zero_coupon_bond_option.sensitivities.prepare_stencils"
    );
}

void prepare_zero_coupon_bond_option_diagonal_sensitivity_stencils_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "ornstein_uhlenbeck.zero_coupon_bond_option.sensitivities.prepare_diagonal_stencils"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders, typename Outputs>
void launch_sensitivities(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ::ai_factory::workbench::monte_carlo::price_gradients::
        DevicePreparedStencilOutputs<
            pg::SensitivityTraits<Orders>::node_capacity
        > stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    Outputs outputs
) {
    if (configuration.method != pg::PricingMethod::closed_form) {
        throw std::invalid_argument(
            "ornstein_uhlenbeck.zero_coupon_bond_option sensitivities require closed-form pricing."
        );
    }
    using PricingPolicy = ::ai_factory::workbench::fixed_income::StandaloneZeroCouponBondOptionClosedFormPricingPolicy<
        ModelParameters, Side
    >;
    using Policy = ::ai_factory::workbench::fixed_income::price_gradients::
    ScalarScenarioClosedFormPolicy<
        PricingPolicy, ModelParameters, product::ZeroCouponBondOptionParameters
    >;
    closed_form::price_gradients::launch_device_prepared<Orders, Policy>(
        host,
        device,
        stencil_outputs,
        configuration,
        pg::as_sensitivity_outputs(outputs),
        "ornstein_uhlenbeck.zero_coupon_bond_option.sensitivities.closed_form",
        option_side_name(Side)
    );
}

template<OptionSide Side>
void launch_ornstein_uhlenbeck_zero_coupon_bond_option_price_gradients_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    launch_sensitivities<Side, pg::SensitivityOrders::first>(
        host, device, stencil_outputs, configuration, outputs
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_ornstein_uhlenbeck_zero_coupon_bond_option_diagonal_sensitivities_cuda(
    const ZeroCouponBondOptionPriceGradientPlan& host,
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    launch_sensitivities<Side, Orders>(
        host, device, stencil_outputs, configuration, outputs
    );
}

#define AI_FACTORY_INSTANTIATE(side)                                        \
template void launch_ornstein_uhlenbeck_zero_coupon_bond_option_price_gradients_cuda<    \
    side>(const ZeroCouponBondOptionPriceGradientPlan&,                           \
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs,                          \
    ZeroCouponBondOptionPriceGradientPlan::StencilOutputs,                        \
    const pg::LaunchConfiguration&, pg::Outputs);                            \
template void launch_ornstein_uhlenbeck_zero_coupon_bond_option_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::second>(                                   \
    const ZeroCouponBondOptionPriceGradientPlan&,                                 \
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs,                          \
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs,                \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);                 \
template void launch_ornstein_uhlenbeck_zero_coupon_bond_option_diagonal_sensitivities_cuda< \
    side, pg::SensitivityOrders::first_and_second>(                         \
    const ZeroCouponBondOptionPriceGradientPlan&,                                 \
    ZeroCouponBondOptionPriceGradientPlan::DeviceInputs,                          \
    ZeroCouponBondOptionPriceGradientPlan::DiagonalStencilOutputs,                \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs)

AI_FACTORY_INSTANTIATE(OptionSide::call);
AI_FACTORY_INSTANTIATE(OptionSide::put);
#undef AI_FACTORY_INSTANTIATE

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
