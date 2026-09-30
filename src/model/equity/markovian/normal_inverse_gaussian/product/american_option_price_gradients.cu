// Generated normal_inverse_gaussian composition over central LSM and frozen-exercise replay.
#include "model/equity/markovian/normal_inverse_gaussian/product/american_option_price_gradients.cuh"

#include "common/equity/price_gradients/device_prepared_stencil_launcher.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_exact_transition_frozen_exercise_replay.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_launcher.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dynamics_impl.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/price_gradients/coupled_dynamics_impl.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_gradients/device_prepared_frozen_replay_node_graph.cuh"
#include "product/american_option/price_gradients/device_prepared_frozen_replay_mixed_node_graph.cuh"
#include "product/american_option/price_gradients/device_prepared_pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {
namespace {

namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

using Schedule = simulation::ExactTransitionMaturityAlignedExerciseSchedule<
    normal_inverse_gaussian::DynamicsPolicy
>;
using Continuation = product::SpotLogMoneynessContinuationState<normal_inverse_gaussian::DynamicsPolicy>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis,
    longstaff_schwartz::RegressionRefinement::none
>;
template<OptionSide Side, pg::SensitivityOrders Orders>
using Replay = lspg::DevicePreparedExactTransitionFrozenExerciseReplay<
    mpg::CoupledDynamics,
    pg::SensitivityTraits<Orders>::node_capacity
>;
template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay = lspg::FrozenExerciseTimeReplay
>
using Policy = product::AmericanOptionDevicePreparedSensitivityPolicy<
    Schedule,
    Side,
    Continuation,
    Replay<Side, Orders>,
    typename AmericanOptionPriceGradientPlan::Preparation,
    Orders,
    Regressor,
    ExerciseReplay
>;
constexpr std::size_t kNodeGraphMaximumSensitivities = 7U;
constexpr unsigned int kNodeGraphGroupSize = 16U;
constexpr unsigned int kNodeGraphNodesPerWorker = 2U;
using NodeGraphTuning = mcpg::tuning::DefaultTerminalNodeTuning;
constexpr std::size_t kMixedNodeGraphMaximumSensitivities = 7U;
constexpr std::size_t kMixedNodeGraphMaximumPairs = 21U;
constexpr unsigned int kMixedNodeGraphTeamSize = 64U;
constexpr unsigned int kMixedNodeGraphNodesPerWorker = 2U;
template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay = lspg::FrozenExerciseTimeReplay
>
using NodeGraphPolicy = american_option_pg::FrozenReplayNodeGraphPolicy<
    Policy<Side, Orders, ExerciseReplay>,
    kNodeGraphMaximumSensitivities,
    kNodeGraphGroupSize,
    kNodeGraphNodesPerWorker,
    NodeGraphTuning
>;
template<
    OptionSide Side,
    typename ExerciseReplay = lspg::FrozenExerciseTimeReplay
>
using MixedNodeGraphPolicy =
    american_option_pg::FrozenReplayMixedNodeGraphPolicy<
        Policy<
            Side, pg::SensitivityOrders::first_and_second, ExerciseReplay
        >,
        kMixedNodeGraphMaximumSensitivities,
        kMixedNodeGraphMaximumPairs,
        kMixedNodeGraphTeamSize,
        kMixedNodeGraphNodesPerWorker,
        NodeGraphTuning
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
        "normal_inverse_gaussian.american_option.price_gradients.stencil_preparation"
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
        "normal_inverse_gaussian.american_option.diagonal.stencil_preparation"
    );
}

namespace {

template<OptionSide Side, typename ExerciseReplay>
longstaff_schwartz::LaunchResult
launch_american_option_price_gradients_for_replay(
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
        Policy<Side, pg::SensitivityOrders::first, ExerciseReplay>,
        Regressor
    >(
        host,
        device,
        stencil_outputs,
        launch,
        sensitivity_outputs,
        "normal_inverse_gaussian.american_option.sensitivities",
        Side == OptionSide::call ? "call/nodes=3/B=1" : "put/nodes=3/B=1",
        "normal_inverse_gaussian American sensitivities"
    );
}

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay
>
longstaff_schwartz::LaunchResult
launch_american_option_diagonal_sensitivities_for_replay(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    return lspg::launch_device_prepared_sensitivities<
        Orders,
        Policy<Side, Orders, ExerciseReplay>,
        Regressor
    >(
        host,
        device,
        stencil_outputs,
        launch,
        outputs,
        "normal_inverse_gaussian.american_option.sensitivities",
        Side == OptionSide::call ? "call/nodes=4/B=1" : "put/nodes=4/B=1",
        "normal_inverse_gaussian American sensitivities"
    );
}


}  // namespace

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_price_gradients_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs
) {
    return launch_american_option_price_gradients_for_replay<
        Side, lspg::FrozenExerciseTimeReplay
    >(host, device, stencil_outputs, launch, outputs);
}

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_price_gradients_with_replay_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return launch_american_option_price_gradients_for_replay<
                Side, ExerciseReplay
            >(host, device, stencil_outputs, launch, outputs);
        }
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_diagonal_sensitivities_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
) {
    return launch_american_option_diagonal_sensitivities_for_replay<
        Side, Orders, lspg::FrozenExerciseTimeReplay
    >(host, device, stencil_outputs, launch, outputs);
}

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_diagonal_sensitivities_with_replay_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    lspg::ExerciseReplayStrategy replay
) {
    static_assert(pg::requests_second_v<Orders>);
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return launch_american_option_diagonal_sensitivities_for_replay<
                Side, Orders, ExerciseReplay
            >(host, device, stencil_outputs, launch, outputs);
        }
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t normal_inverse_gaussian_american_option_node_graph_workspace_bytes(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    static_assert(pg::requests_second_v<Orders>);
    return lspg::frozen_replay_node_graph_workspace_bytes<
        Orders,
        kNodeGraphMaximumSensitivities,
        kNodeGraphGroupSize,
        kNodeGraphNodesPerWorker,
        NodeGraphTuning
    >(host, configuration);
}

template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t normal_inverse_gaussian_american_option_node_graph_workspace_bytes_with_replay(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            static_cast<void>(sizeof(ExerciseReplay));
            return normal_inverse_gaussian_american_option_node_graph_workspace_bytes<
                Side, Orders
            >(host, configuration);
        }
    );
}

namespace {

template<
    OptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay
>
longstaff_schwartz::LaunchResult
launch_american_option_node_graph_for_replay(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    static_assert(pg::requests_second_v<Orders>);
    return lspg::launch_device_prepared_node_graph_sensitivities<
        Orders,
        NodeGraphPolicy<Side, Orders, ExerciseReplay>,
        Regressor,
        kNodeGraphMaximumSensitivities,
        kNodeGraphGroupSize,
        kNodeGraphNodesPerWorker,
        NodeGraphTuning
    >(
        host,
        device,
        stencil_outputs,
        launch,
        outputs,
        workspace,
        workspace_bytes,
        "normal_inverse_gaussian.american_option.sensitivities",
        Side == OptionSide::call
            ? "call/nodes=graph" : "put/nodes=graph",
        "normal_inverse_gaussian American node-graph sensitivities"
    );
}

}  // namespace

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_node_graph_sensitivities_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    return launch_american_option_node_graph_for_replay<
        Side, Orders, lspg::FrozenExerciseTimeReplay
    >(
        host, device, stencil_outputs, launch, outputs,
        workspace, workspace_bytes
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_node_graph_sensitivities_with_replay_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return launch_american_option_node_graph_for_replay<
                Side, Orders, ExerciseReplay
            >(
                host, device, stencil_outputs, launch, outputs,
                workspace, workspace_bytes
            );
        }
    );
}

namespace {

template<OptionSide Side, typename ExerciseReplay>
std::size_t american_option_mixed_node_graph_workspace_bytes_for_replay(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return lspg::frozen_replay_mixed_node_graph_workspace_bytes<
        MixedNodeGraphPolicy<Side, ExerciseReplay>,
        kMixedNodeGraphMaximumSensitivities,
        kMixedNodeGraphMaximumPairs,
        kMixedNodeGraphTeamSize,
        kMixedNodeGraphNodesPerWorker,
        NodeGraphTuning
    >(host, configuration);
}

}  // namespace

template<OptionSide Side>
std::size_t normal_inverse_gaussian_american_option_mixed_node_graph_workspace_bytes(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return american_option_mixed_node_graph_workspace_bytes_for_replay<
        Side, lspg::FrozenExerciseTimeReplay
    >(host, configuration);
}

template<OptionSide Side>
std::size_t normal_inverse_gaussian_american_option_mixed_node_graph_workspace_bytes_with_replay(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return american_option_mixed_node_graph_workspace_bytes_for_replay<
                Side, ExerciseReplay
            >(host, configuration);
        }
    );
}

namespace {

template<OptionSide Side, typename ExerciseReplay>
longstaff_schwartz::LaunchResult
launch_american_option_mixed_node_graph_for_replay(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    AmericanOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    return lspg::launch_device_prepared_mixed_node_graph_sensitivities<
        MixedNodeGraphPolicy<Side, ExerciseReplay>,
        Regressor,
        kMixedNodeGraphMaximumSensitivities,
        kMixedNodeGraphMaximumPairs,
        kMixedNodeGraphTeamSize,
        kMixedNodeGraphNodesPerWorker,
        NodeGraphTuning
    >(
        host,
        device,
        stencil_outputs,
        mixed_stencil_outputs,
        launch,
        outputs,
        mixed_outputs,
        workspace,
        workspace_bytes,
        "normal_inverse_gaussian.american_option.sensitivities.mixed_node_graph",
        Side == OptionSide::call
            ? "call/nodes=mixed_graph" : "put/nodes=mixed_graph",
        "normal_inverse_gaussian American mixed node-graph sensitivities"
    );
}

}  // namespace

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_mixed_node_graph_sensitivities_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    AmericanOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    return launch_american_option_mixed_node_graph_for_replay<
        Side, lspg::FrozenExerciseTimeReplay
    >(
        host, device, stencil_outputs, mixed_stencil_outputs,
        launch, outputs, mixed_outputs, workspace, workspace_bytes
    );
}

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_normal_inverse_gaussian_american_option_mixed_node_graph_sensitivities_with_replay_cuda(
    const AmericanOptionPriceGradientPlan& host,
    AmericanOptionPriceGradientPlan::DeviceInputs device,
    AmericanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    AmericanOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return launch_american_option_mixed_node_graph_for_replay<
                Side, ExerciseReplay
            >(
                host, device, stencil_outputs, mixed_stencil_outputs,
                launch, outputs, mixed_outputs, workspace, workspace_bytes
            );
        }
    );
}

#define AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(SIDE) \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_price_gradients_cuda<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::StencilOutputs, \
        const pg::LaunchConfiguration&, pg::Outputs); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_price_gradients_with_replay_cuda<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::StencilOutputs, \
        const pg::LaunchConfiguration&, pg::Outputs, \
        lspg::ExerciseReplayStrategy); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_diagonal_sensitivities_with_replay_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        lspg::ExerciseReplayStrategy); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_diagonal_sensitivities_with_replay_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        lspg::ExerciseReplayStrategy); \
    template std::size_t \
    normal_inverse_gaussian_american_option_node_graph_workspace_bytes< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&); \
    template std::size_t \
    normal_inverse_gaussian_american_option_node_graph_workspace_bytes< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&); \
    template std::size_t \
    normal_inverse_gaussian_american_option_node_graph_workspace_bytes_with_replay< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&, lspg::ExerciseReplayStrategy); \
    template std::size_t \
    normal_inverse_gaussian_american_option_node_graph_workspace_bytes_with_replay< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&, lspg::ExerciseReplayStrategy); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_node_graph_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        void*, std::size_t); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_node_graph_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        void*, std::size_t); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_node_graph_sensitivities_with_replay_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        void*, std::size_t, lspg::ExerciseReplayStrategy); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_node_graph_sensitivities_with_replay_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        void*, std::size_t, lspg::ExerciseReplayStrategy); \
    template std::size_t \
    normal_inverse_gaussian_american_option_mixed_node_graph_workspace_bytes<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&); \
    template std::size_t \
    normal_inverse_gaussian_american_option_mixed_node_graph_workspace_bytes_with_replay<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&, lspg::ExerciseReplayStrategy); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_mixed_node_graph_sensitivities_cuda<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        AmericanOptionPriceGradientPlan::MixedStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        pg::MixedSensitivityOutputs, void*, std::size_t); \
    template longstaff_schwartz::LaunchResult \
    launch_normal_inverse_gaussian_american_option_mixed_node_graph_sensitivities_with_replay_cuda<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        AmericanOptionPriceGradientPlan::MixedStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        pg::MixedSensitivityOutputs, void*, std::size_t, \
        lspg::ExerciseReplayStrategy)

AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(OptionSide::call);
AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(OptionSide::put);

#undef AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
