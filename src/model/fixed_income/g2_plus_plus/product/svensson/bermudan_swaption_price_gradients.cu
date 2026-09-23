// Generated G2++/Svensson Bermudan frozen-policy replay.
#include "model/fixed_income/g2_plus_plus/product/svensson/bermudan_swaption_price_gradients.cuh"

#include "common/fixed_income/bermudan_swaption_continuation_state.cuh"
#include "common/fixed_income/scalar_rate_continuation_state.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_launcher.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "common/simulation/early_exercise_schedule.cuh"
#include "common/simulation/terminal_forward_exercise_schedule.cuh"
#include "model/fixed_income/g2_plus_plus/dynamics.cuh"
#include "model/fixed_income/g2_plus_plus/svensson/analytics_impl.cuh"
#include "product/bermudan_swaption/price_gradients/device_prepared_pricing_policy.cuh"
#include "product/bermudan_swaption/pricing_policy.cuh"
#include "product/bermudan_swaption/terminal_forward_pricing_policy.cuh"

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson {
namespace {

using Dynamics = g2_plus_plus::joint::DynamicsPolicy;
using Schedule = simulation::ExactTransitionRegularExerciseSchedule<Dynamics>;
using Analytics = BermudanSwaptionAnalyticsPolicy;
using ContinuationState = ::ai_factory::workbench::fixed_income::TwoFactorRateContinuationState<Dynamics>;
using CurveParameters = curve::svensson::SvenssonParameters;
template<SwaptionSide Side>
using CentralPolicy = product::FittedBermudanSwaptionPricingPolicy<
    Schedule, CurveParameters, Analytics, Side, ContinuationState
>;

template<SwaptionSide Side, pg::SensitivityOrders Orders>
using Policy = product::BermudanSwaptionDevicePreparedSensitivityPolicy<
    CentralPolicy<Side>,
    typename BermudanSwaptionPriceGradientPlan::Preparation,
    typename BermudanSwaptionPriceGradientPlan::DeviceInputs,
    Orders,
    false
>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::TwoFactorHermiteBasis
>;

}  // namespace

void prepare_bermudan_swaption_price_gradient_stencils_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "g2_plus_plus.svensson.bermudan_swaption.price_gradients.prepare_stencils"
    );
}

void prepare_bermudan_swaption_diagonal_sensitivity_stencils_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
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
        "g2_plus_plus.svensson.bermudan_swaption.diagonal.prepare_stencils"
    );
}

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_g2_plus_plus_svensson_bermudan_swaption_price_gradients_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs
) {
    const pg::SensitivityOutputs sensitivity_outputs{
        outputs.prices,
        outputs.price_standard_errors,
        outputs.gradients,
        outputs.gradient_standard_errors,
        nullptr,
        nullptr,
        outputs.price_capacity,
        outputs.gradient_capacity,
    };
    return longstaff_schwartz::price_gradients::
        launch_device_prepared_sensitivities<
            pg::SensitivityOrders::first,
            Policy<Side, pg::SensitivityOrders::first>,
            Regressor
        >(
            host,
            device,
            stencil_outputs,
            launch,
            sensitivity_outputs,
            "g2_plus_plus.svensson.bermudan_swaption.sensitivities",
            Side == SwaptionSide::payer
                ? "payer/nodes=3/B=1" : "receiver/nodes=3/B=1",
            "G2++/Svensson Bermudan sensitivities"
        );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_g2_plus_plus_svensson_bermudan_swaption_diagonal_sensitivities_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    return longstaff_schwartz::price_gradients::
        launch_device_prepared_sensitivities<Orders, Policy<Side, Orders>, Regressor>(
            host,
            device,
            stencil_outputs,
            launch,
            outputs,
            "g2_plus_plus.svensson.bermudan_swaption.sensitivities",
            Side == SwaptionSide::payer
                ? "payer/nodes=4/B=1" : "receiver/nodes=4/B=1",
            "G2++/Svensson Bermudan sensitivities"
        );
}

#define AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SIDE)                 \
    template longstaff_schwartz::LaunchResult                               \
    launch_g2_plus_plus_svensson_bermudan_swaption_price_gradients_cuda<SIDE>( \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::StencilOutputs,                  \
        const pg::LaunchConfiguration&, pg::Outputs);                       \
    template longstaff_schwartz::LaunchResult                               \
    launch_g2_plus_plus_svensson_bermudan_swaption_diagonal_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs);            \
    template longstaff_schwartz::LaunchResult                               \
    launch_g2_plus_plus_svensson_bermudan_swaption_diagonal_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs)

AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SwaptionSide::payer);
AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SwaptionSide::receiver);

#undef AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson
