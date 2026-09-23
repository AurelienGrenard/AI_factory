// Generated bates composition over central LSM and frozen-exercise replay.
#include "model/equity/markovian/bates/product/american_option_price_gradients.cuh"

#include "common/equity/price_gradients/device_prepared_stencil_launcher.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_fixed_step_frozen_exercise_replay.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_launcher.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/equity/markovian/bates/dynamics_impl.cuh"
#include "model/equity/markovian/bates/price_gradients/coupled_dynamics_impl.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_gradients/device_prepared_pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::bates {
namespace {

using Schedule = simulation::FixedStepMaturityAlignedExerciseSchedule<
    bates::DynamicsPolicy
>;
using Continuation = product::SpotAndScaledStateContinuationState<bates::DynamicsPolicy, &bates::State::variance, &bates::ModelParameters::theta>;
template<OptionSide Side, pg::SensitivityOrders Orders>
using Replay = lspg::DevicePreparedFixedStepFrozenExerciseReplay<
    mpg::CoupledDynamics,
    pg::SensitivityTraits<Orders>::node_capacity
>;
template<OptionSide Side, pg::SensitivityOrders Orders>
using Policy = product::AmericanOptionDevicePreparedSensitivityPolicy<
    Schedule,
    Side,
    Continuation,
    Replay<Side, Orders>,
    typename AmericanOptionPriceGradientPlan::Preparation,
    Orders
>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis,
    longstaff_schwartz::RegressionRefinement::none
>;

}  // namespace

void prepare_american_option_price_gradient_stencils_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    ::ai_factory::workbench::equity::price_gradients::
        prepare_device_sensitivity_stencils<
        pg::SensitivityOrders::first
    >(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "bates.american_option.price_gradients.stencil_preparation"
    );
}

void prepare_american_option_diagonal_sensitivity_stencils_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    ::ai_factory::workbench::equity::price_gradients::
        prepare_device_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "bates.american_option.diagonal.stencil_preparation"
    );
}

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_bates_american_option_price_gradients_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
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
    return lspg::launch_device_prepared_sensitivities<
        pg::SensitivityOrders::first,
        Policy<Side, pg::SensitivityOrders::first>,
        Regressor
    >(
        host,
        device,
        stencil_outputs,
        launch,
        sensitivity_outputs,
        "bates.american_option.sensitivities",
        Side == OptionSide::call ? "call/nodes=3/B=1" : "put/nodes=3/B=1",
        "bates American sensitivities"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_bates_american_option_diagonal_sensitivities_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    return lspg::launch_device_prepared_sensitivities<
        Orders,
        Policy<Side, Orders>,
        Regressor
    >(
        host,
        device,
        stencil_outputs,
        launch,
        outputs,
        "bates.american_option.sensitivities",
        Side == OptionSide::call ? "call/nodes=4/B=1" : "put/nodes=4/B=1",
        "bates American sensitivities"
    );
}

#define AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(SIDE) \
    template longstaff_schwartz::LaunchResult \
    launch_bates_american_option_price_gradients_cuda<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::StencilOutputs, \
        const pg::LaunchConfiguration&, pg::Outputs); \
    template longstaff_schwartz::LaunchResult \
    launch_bates_american_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template longstaff_schwartz::LaunchResult \
    launch_bates_american_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs)

AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(OptionSide::call);
AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(OptionSide::put);

#undef AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES

}  // namespace ai_factory::workbench::model::equity::bates
