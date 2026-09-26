// Generated ${model} composition over central LSM and frozen-exercise replay.
#include "model/equity/markovian/${model}/product/american_option_price_gradients.cuh"

#include "common/equity/price_gradients/device_prepared_stencil_launcher.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/price_gradients/${lsm_replay_header}"
#include "common/longstaff_schwartz/price_gradients/device_prepared_launcher.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/equity/markovian/${model}/dynamics_impl.cuh"
#include "model/equity/markovian/${model}/price_gradients/coupled_dynamics_impl.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_gradients/device_prepared_frozen_exercise_node_graph.cuh"
#include "product/american_option/price_gradients/device_prepared_frozen_exercise_mixed_node_graph.cuh"
#include "product/american_option/price_gradients/device_prepared_pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::${model} {
namespace {

namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

using Schedule = simulation::${lsm_schedule}<
    ${model}::DynamicsPolicy
>;
using Continuation = ${lsm_continuation};
template<OptionSide Side, pg::SensitivityOrders Orders>
using Replay = lspg::${lsm_replay}<
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
constexpr std::size_t kNodeGraphMaximumSensitivities = ${maximum}U;
constexpr unsigned int kNodeGraphGroupSize = ${graph_group_size}U;
constexpr unsigned int kNodeGraphNodesPerWorker = ${graph_nodes_per_worker}U;
using NodeGraphTuning = mcpg::tuning::DefaultTerminalNodeTuning;
constexpr std::size_t kMixedNodeGraphMaximumSensitivities = ${maximum}U;
constexpr std::size_t kMixedNodeGraphMaximumPairs = ${mixed_maximum}U;
constexpr unsigned int kMixedNodeGraphTeamSize = ${mixed_team_size}U;
constexpr unsigned int kMixedNodeGraphNodesPerWorker = ${mixed_nodes_per_worker}U;
template<OptionSide Side, pg::SensitivityOrders Orders>
using NodeGraphPolicy = american_option_pg::FrozenExerciseNodeGraphPolicy<
    Policy<Side, Orders>,
    kNodeGraphMaximumSensitivities,
    kNodeGraphGroupSize,
    kNodeGraphNodesPerWorker,
    NodeGraphTuning
>;
template<OptionSide Side>
using MixedNodeGraphPolicy =
    american_option_pg::FrozenExerciseMixedNodeGraphPolicy<
        Policy<Side, pg::SensitivityOrders::first_and_second>,
        kMixedNodeGraphMaximumSensitivities,
        kMixedNodeGraphMaximumPairs,
        kMixedNodeGraphTeamSize,
        kMixedNodeGraphNodesPerWorker,
        NodeGraphTuning
    >;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis,
    longstaff_schwartz::RegressionRefinement::${lsm_refinement}
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
        "${model}.american_option.price_gradients.stencil_preparation"
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
        "${model}.american_option.diagonal.stencil_preparation"
    );
}

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_${model}_american_option_price_gradients_cuda(
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
        "${model}.american_option.sensitivities",
        Side == OptionSide::call ? "call/nodes=3/B=1" : "put/nodes=3/B=1",
        "${model} American sensitivities"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_${model}_american_option_diagonal_sensitivities_cuda(
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
        "${model}.american_option.sensitivities",
        Side == OptionSide::call ? "call/nodes=4/B=1" : "put/nodes=4/B=1",
        "${model} American sensitivities"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
std::size_t ${model}_american_option_node_graph_workspace_bytes(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    static_assert(pg::requests_second_v<Orders>);
    return lspg::frozen_exercise_node_graph_workspace_bytes<
        Orders,
        kNodeGraphMaximumSensitivities,
        kNodeGraphGroupSize,
        kNodeGraphNodesPerWorker,
        NodeGraphTuning
    >(host, configuration);
}

template<OptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_${model}_american_option_node_graph_sensitivities_cuda(
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
        NodeGraphPolicy<Side, Orders>,
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
        "${model}.american_option.sensitivities",
        Side == OptionSide::call
            ? "call/nodes=graph" : "put/nodes=graph",
        "${model} American node-graph sensitivities"
    );
}


template<OptionSide Side>
std::size_t ${model}_american_option_mixed_node_graph_workspace_bytes(
    const AmericanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return lspg::frozen_exercise_mixed_node_graph_workspace_bytes<
        MixedNodeGraphPolicy<Side>,
        kMixedNodeGraphMaximumSensitivities,
        kMixedNodeGraphMaximumPairs,
        kMixedNodeGraphTeamSize,
        kMixedNodeGraphNodesPerWorker,
        NodeGraphTuning
    >(host, configuration);
}

template<OptionSide Side>
longstaff_schwartz::LaunchResult
launch_${model}_american_option_mixed_node_graph_sensitivities_cuda(
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
        MixedNodeGraphPolicy<Side>,
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
        "${model}.american_option.sensitivities.mixed_node_graph",
        Side == OptionSide::call
            ? "call/nodes=mixed_graph" : "put/nodes=mixed_graph",
        "${model} American mixed node-graph sensitivities"
    );
}

#define AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(SIDE) \
    template longstaff_schwartz::LaunchResult \
    launch_${model}_american_option_price_gradients_cuda<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::StencilOutputs, \
        const pg::LaunchConfiguration&, pg::Outputs); \
    template longstaff_schwartz::LaunchResult \
    launch_${model}_american_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template longstaff_schwartz::LaunchResult \
    launch_${model}_american_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template std::size_t \
    ${model}_american_option_node_graph_workspace_bytes< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&); \
    template std::size_t \
    ${model}_american_option_node_graph_workspace_bytes< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&); \
    template longstaff_schwartz::LaunchResult \
    launch_${model}_american_option_node_graph_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        void*, std::size_t); \
    template longstaff_schwartz::LaunchResult \
    launch_${model}_american_option_node_graph_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        void*, std::size_t); \
    template std::size_t \
    ${model}_american_option_mixed_node_graph_workspace_bytes<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&); \
    template longstaff_schwartz::LaunchResult \
    launch_${model}_american_option_mixed_node_graph_sensitivities_cuda<SIDE>( \
        const AmericanOptionPriceGradientPlan&, \
        AmericanOptionPriceGradientPlan::DeviceInputs, \
        AmericanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        AmericanOptionPriceGradientPlan::MixedStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        pg::MixedSensitivityOutputs, void*, std::size_t)

AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(OptionSide::call);
AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES(OptionSide::put);

#undef AI_FACTORY_INSTANTIATE_AMERICAN_SENSITIVITIES

}  // namespace ai_factory::workbench::model::equity::${model}
