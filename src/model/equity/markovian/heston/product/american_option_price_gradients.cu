// Heston composition over central LSM and coupled frozen-exercise replay.
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"

#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/heston/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/heston/price_gradients/frozen_exercise_replay.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_gradients/pricing_policy.cuh"

#include <stdexcept>

namespace ai_factory::workbench::model::equity::heston {
namespace {

using Schedule = simulation::FixedStepMaturityAlignedExerciseSchedule<
    heston::DynamicsPolicy
>;
using Continuation = product::SpotAndScaledStateContinuationState<
    heston::DynamicsPolicy,
    &heston::State::variance,
    &heston::ModelParameters::theta
>;
template<OptionSide Side>
using Policy = product::AmericanOptionPriceGradientPolicy<
    Schedule,
    Side,
    Continuation,
    price_gradients::FrozenExerciseReplay
>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis,
    longstaff_schwartz::RegressionRefinement::none
>;

}  // namespace

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_heston_american_option_price_gradients_cuda(
    const AmericanOptionPriceGradientPlan& host,
    ::ai_factory::workbench::price_gradients::DeviceInputs<
        AmericanOptionPriceGradientPlan::ScenarioType
    > device,
    const ::ai_factory::workbench::price_gradients::LaunchConfiguration& launch,
    ::ai_factory::workbench::price_gradients::Outputs outputs
) {
    namespace pg = ::ai_factory::workbench::price_gradients;
    pg::validate_plan_buffers(host, device, launch, outputs);
    if (launch.method != pg::PricingMethod::monte_carlo) {
        throw std::invalid_argument(
            "Heston American gradients require Monte Carlo."
        );
    }
    if (launch.sensitivity_batch_size != 1U) {
        throw std::invalid_argument(
            "Heston American frozen gradients currently require B=1."
        );
    }

    using GradientPolicy = Policy<Side>;
    const std::size_t sensitivity_count = host.sensitivity_count();
    const std::size_t scenarios_per_row = 1U + 2U * sensitivity_count;
    const std::size_t result_offset = launch.result_offset;
    const std::size_t result_count = launch.result_count;
    validate_row_seed_range(host.result_count, launch.base_seed);
    const typename GradientPolicy::HostInputs host_inputs{
        host.scenarios.data() + result_offset * scenarios_per_row,
        result_count * scenarios_per_row,
        sensitivity_count == 0U
            ? nullptr
            : host.stencils.data() + result_offset * sensitivity_count,
        result_count * sensitivity_count,
        sensitivity_count,
    };
    const pg::DeviceInputs<AmericanOptionPriceGradientPlan::ScenarioType>
        batch_device{
            device.scenarios + result_offset * scenarios_per_row,
            device.scenario_capacity - result_offset * scenarios_per_row,
            sensitivity_count == 0U
                ? nullptr
                : device.stencils + result_offset * sensitivity_count,
            device.stencil_capacity - result_offset * sensitivity_count,
        };
    const typename GradientPolicy::DeviceInputs device_inputs{
        batch_device,
        sensitivity_count,
        result_offset,
        outputs.gradients,
        outputs.gradient_standard_errors,
    };
    return longstaff_schwartz::launch_longstaff_schwartz_cuda<
        GradientPolicy, Regressor
    >(
        device_inputs,
        host_inputs,
        result_count,
        launch.paths_per_price,
        simulation::FixedStepTimeConfiguration{
            host.time.dt,
            host.time.simulation_steps_per_day,
        },
        launch.threads_per_block,
        launch.block_count,
        launch.base_seed + result_offset,
        outputs.prices,
        outputs.price_standard_errors,
        "heston.american_option_price_gradients",
        option_side_name(Side),
        "Heston American price-gradients"
    );
}

template longstaff_schwartz::LaunchResult
launch_heston_american_option_price_gradients_cuda<OptionSide::call>(
    const AmericanOptionPriceGradientPlan&,
    ::ai_factory::workbench::price_gradients::DeviceInputs<
        AmericanOptionPriceGradientPlan::ScenarioType
    >,
    const ::ai_factory::workbench::price_gradients::LaunchConfiguration&,
    ::ai_factory::workbench::price_gradients::Outputs
);
template longstaff_schwartz::LaunchResult
launch_heston_american_option_price_gradients_cuda<OptionSide::put>(
    const AmericanOptionPriceGradientPlan&,
    ::ai_factory::workbench::price_gradients::DeviceInputs<
        AmericanOptionPriceGradientPlan::ScenarioType
    >,
    const ::ai_factory::workbench::price_gradients::LaunchConfiguration&,
    ::ai_factory::workbench::price_gradients::Outputs
);

}  // namespace ai_factory::workbench::model::equity::heston
