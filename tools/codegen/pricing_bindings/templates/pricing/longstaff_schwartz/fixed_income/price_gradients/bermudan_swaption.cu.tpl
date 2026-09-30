// Generated ${model_display}${curve_display_suffix} Bermudan frozen-policy replay.
#include "${unit_path}.cuh"

#include "common/fixed_income/bermudan_swaption_continuation_state.cuh"
#include "common/fixed_income/scalar_rate_continuation_state.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_launcher.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "common/simulation/early_exercise_schedule.cuh"
#include "common/simulation/terminal_forward_exercise_schedule.cuh"
${implementation_includes}#include "product/bermudan_swaption/price_gradients/device_prepared_frozen_replay_node_graph.cuh"
#include "product/bermudan_swaption/price_gradients/device_prepared_frozen_replay_mixed_node_graph.cuh"
#include "product/bermudan_swaption/price_gradients/device_prepared_pricing_policy.cuh"
#include "product/bermudan_swaption/pricing_policy.cuh"
#include "product/bermudan_swaption/terminal_forward_pricing_policy.cuh"

namespace ai_factory::workbench::model::fixed_income::${binding_namespace} {
namespace {

namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
namespace lspg =
    ::ai_factory::workbench::longstaff_schwartz::price_gradients;

using Dynamics = ${dynamics};
using Schedule = ${schedule};
using Analytics = BermudanSwaptionAnalyticsPolicy;
using ContinuationState = ${continuation};
${curve_type_alias}template<SwaptionSide Side>
using CentralPolicy = ${central_policy};
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    ${regressor_basis}
>;

template<
    SwaptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay = lspg::FrozenExerciseTimeReplay
>
using Policy = product::BermudanSwaptionDevicePreparedSensitivityPolicy<
    CentralPolicy<Side>,
    typename BermudanSwaptionPriceGradientPlan::Preparation,
    typename BermudanSwaptionPriceGradientPlan::DeviceInputs,
    Orders,
    ${terminal_forward},
    Regressor,
    ExerciseReplay
>;
constexpr std::size_t kNodeGraphMaximumSensitivities = ${maximum}U;
constexpr unsigned int kNodeGraphGroupSize = ${graph_group_size}U;
constexpr unsigned int kNodeGraphNodesPerWorker = ${graph_nodes_per_worker}U;
using NodeGraphTuning = mcpg::tuning::DefaultTerminalNodeTuning;
constexpr std::size_t kMixedNodeGraphMaximumSensitivities = ${maximum}U;
constexpr std::size_t kMixedNodeGraphMaximumPairs = ${mixed_maximum}U;
constexpr unsigned int kMixedNodeGraphTeamSize = ${mixed_team_size}U;
constexpr unsigned int kMixedNodeGraphNodesPerWorker = ${mixed_nodes_per_worker}U;
template<
    SwaptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay = lspg::FrozenExerciseTimeReplay
>
using NodeGraphPolicy = bermudan_pg::FrozenReplayNodeGraphPolicy<
    Policy<Side, Orders, ExerciseReplay>,
    kNodeGraphMaximumSensitivities,
    kNodeGraphGroupSize,
    kNodeGraphNodesPerWorker,
    NodeGraphTuning
>;
template<
    SwaptionSide Side,
    typename ExerciseReplay = lspg::FrozenExerciseTimeReplay
>
using MixedNodeGraphPolicy =
    bermudan_pg::FrozenReplayMixedNodeGraphPolicy<
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
        "${diagnostic_name}.price_gradients.prepare_stencils"
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
        "${diagnostic_name}.diagonal.prepare_stencils"
    );
}

namespace {

template<SwaptionSide Side, typename ExerciseReplay>
longstaff_schwartz::LaunchResult
launch_bermudan_swaption_price_gradients_for_replay(
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
            Policy<Side, pg::SensitivityOrders::first, ExerciseReplay>,
            Regressor
        >(
            host,
            device,
            stencil_outputs,
            launch,
            sensitivity_outputs,
            "${diagnostic_name}.sensitivities",
            Side == SwaptionSide::payer
                ? "payer/nodes=3/B=1" : "receiver/nodes=3/B=1",
            "${model_display}${curve_display_suffix} Bermudan sensitivities"
        );
}

template<
    SwaptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay
>
longstaff_schwartz::LaunchResult
launch_bermudan_swaption_diagonal_sensitivities_for_replay(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
) {
    static_assert(pg::requests_second_v<Orders>);
    return longstaff_schwartz::price_gradients::
        launch_device_prepared_sensitivities<
            Orders, Policy<Side, Orders, ExerciseReplay>, Regressor
        >(
            host,
            device,
            stencil_outputs,
            launch,
            outputs,
            "${diagnostic_name}.sensitivities",
            Side == SwaptionSide::payer
                ? "payer/nodes=4/B=1" : "receiver/nodes=4/B=1",
            "${model_display}${curve_display_suffix} Bermudan sensitivities"
        );
}

}  // namespace

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_price_gradients_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs
) {
    return launch_bermudan_swaption_price_gradients_for_replay<
        Side, lspg::FrozenExerciseTimeReplay
    >(host, device, stencil_outputs, launch, outputs);
}

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_price_gradients_with_replay_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return launch_bermudan_swaption_price_gradients_for_replay<
                Side, ExerciseReplay
            >(host, device, stencil_outputs, launch, outputs);
        }
    );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_diagonal_sensitivities_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs
) {
    return launch_bermudan_swaption_diagonal_sensitivities_for_replay<
        Side, Orders, lspg::FrozenExerciseTimeReplay
    >(host, device, stencil_outputs, launch, outputs);
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_diagonal_sensitivities_with_replay_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    lspg::ExerciseReplayStrategy replay
) {
    static_assert(pg::requests_second_v<Orders>);
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return launch_bermudan_swaption_diagonal_sensitivities_for_replay<
                Side, Orders, ExerciseReplay
            >(host, device, stencil_outputs, launch, outputs);
        }
    );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
std::size_t ${function_prefix}_bermudan_swaption_node_graph_workspace_bytes(
    const BermudanSwaptionPriceGradientPlan& host,
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

template<SwaptionSide Side, pg::SensitivityOrders Orders>
std::size_t ${function_prefix}_bermudan_swaption_node_graph_workspace_bytes_with_replay(
    const BermudanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            static_cast<void>(sizeof(ExerciseReplay));
            return ${function_prefix}_bermudan_swaption_node_graph_workspace_bytes<
                Side, Orders
            >(host, configuration);
        }
    );
}

namespace {

template<
    SwaptionSide Side,
    pg::SensitivityOrders Orders,
    typename ExerciseReplay
>
longstaff_schwartz::LaunchResult
launch_bermudan_swaption_node_graph_for_replay(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
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
        "${diagnostic_name}.sensitivities",
        Side == SwaptionSide::payer
            ? "payer/nodes=graph" : "receiver/nodes=graph",
        "${model_display}${curve_display_suffix} Bermudan node-graph sensitivities"
    );
}

}  // namespace

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_node_graph_sensitivities_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    return launch_bermudan_swaption_node_graph_for_replay<
        Side, Orders, lspg::FrozenExerciseTimeReplay
    >(
        host, device, stencil_outputs, launch, outputs,
        workspace, workspace_bytes
    );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_node_graph_sensitivities_with_replay_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return launch_bermudan_swaption_node_graph_for_replay<
                Side, Orders, ExerciseReplay
            >(
                host, device, stencil_outputs, launch, outputs,
                workspace, workspace_bytes
            );
        }
    );
}

namespace {

template<SwaptionSide Side, typename ExerciseReplay>
std::size_t bermudan_swaption_mixed_node_graph_workspace_bytes_for_replay(
    const BermudanSwaptionPriceGradientPlan& host,
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

template<SwaptionSide Side>
std::size_t ${function_prefix}_bermudan_swaption_mixed_node_graph_workspace_bytes(
    const BermudanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    return bermudan_swaption_mixed_node_graph_workspace_bytes_for_replay<
        Side, lspg::FrozenExerciseTimeReplay
    >(host, configuration);
}

template<SwaptionSide Side>
std::size_t ${function_prefix}_bermudan_swaption_mixed_node_graph_workspace_bytes_with_replay(
    const BermudanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration,
    lspg::ExerciseReplayStrategy replay
) {
    return lspg::dispatch_exercise_replay<Regressor>(
        replay,
        [&]<typename ExerciseReplay>() {
            return bermudan_swaption_mixed_node_graph_workspace_bytes_for_replay<
                Side, ExerciseReplay
            >(host, configuration);
        }
    );
}

namespace {

template<SwaptionSide Side, typename ExerciseReplay>
longstaff_schwartz::LaunchResult
launch_bermudan_swaption_mixed_node_graph_for_replay(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    BermudanSwaptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
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
        "${diagnostic_name}.sensitivities.mixed_node_graph",
        Side == SwaptionSide::payer
            ? "payer/nodes=mixed_graph" : "receiver/nodes=mixed_graph",
        "${model_display}${curve_display_suffix} Bermudan mixed node-graph sensitivities"
    );
}

}  // namespace

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_mixed_node_graph_sensitivities_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    BermudanSwaptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    return launch_bermudan_swaption_mixed_node_graph_for_replay<
        Side, lspg::FrozenExerciseTimeReplay
    >(
        host, device, stencil_outputs, mixed_stencil_outputs,
        launch, outputs, mixed_outputs, workspace, workspace_bytes
    );
}

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_${function_prefix}_bermudan_swaption_mixed_node_graph_sensitivities_with_replay_cuda(
    const BermudanSwaptionPriceGradientPlan& host,
    BermudanSwaptionPriceGradientPlan::DeviceInputs device,
    BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    BermudanSwaptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
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
            return launch_bermudan_swaption_mixed_node_graph_for_replay<
                Side, ExerciseReplay
            >(
                host, device, stencil_outputs, mixed_stencil_outputs,
                launch, outputs, mixed_outputs, workspace, workspace_bytes
            );
        }
    );
}

#define AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SIDE)                 \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_price_gradients_cuda<SIDE>( \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::StencilOutputs,                  \
        const pg::LaunchConfiguration&, pg::Outputs);                       \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_price_gradients_with_replay_cuda<SIDE>( \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::StencilOutputs,                  \
        const pg::LaunchConfiguration&, pg::Outputs,                        \
        lspg::ExerciseReplayStrategy);                                      \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_diagonal_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs);            \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_diagonal_sensitivities_with_replay_cuda<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        lspg::ExerciseReplayStrategy);                                      \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_diagonal_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs);            \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_diagonal_sensitivities_with_replay_cuda<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        lspg::ExerciseReplayStrategy);                                      \
    template std::size_t                                                    \
    ${function_prefix}_bermudan_swaption_node_graph_workspace_bytes<        \
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&);                                    \
    template std::size_t                                                    \
    ${function_prefix}_bermudan_swaption_node_graph_workspace_bytes<        \
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&);                                    \
    template std::size_t                                                    \
    ${function_prefix}_bermudan_swaption_node_graph_workspace_bytes_with_replay<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&, lspg::ExerciseReplayStrategy);      \
    template std::size_t                                                    \
    ${function_prefix}_bermudan_swaption_node_graph_workspace_bytes_with_replay<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&, lspg::ExerciseReplayStrategy);      \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_node_graph_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        void*, std::size_t);                                                \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_node_graph_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        void*, std::size_t);                                                \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_node_graph_sensitivities_with_replay_cuda<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        void*, std::size_t, lspg::ExerciseReplayStrategy);                  \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_node_graph_sensitivities_with_replay_cuda<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        void*, std::size_t, lspg::ExerciseReplayStrategy);                  \
    template std::size_t                                                    \
    ${function_prefix}_bermudan_swaption_mixed_node_graph_workspace_bytes<  \
        SIDE>(                                                              \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&);                                    \
    template std::size_t                                                    \
    ${function_prefix}_bermudan_swaption_mixed_node_graph_workspace_bytes_with_replay<\
        SIDE>(                                                              \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&, lspg::ExerciseReplayStrategy);      \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_mixed_node_graph_sensitivities_cuda<\
        SIDE>(                                                              \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        BermudanSwaptionPriceGradientPlan::MixedStencilOutputs,             \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        pg::MixedSensitivityOutputs, void*, std::size_t);                   \
    template longstaff_schwartz::LaunchResult                               \
    launch_${function_prefix}_bermudan_swaption_mixed_node_graph_sensitivities_with_replay_cuda<\
        SIDE>(                                                              \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        BermudanSwaptionPriceGradientPlan::MixedStencilOutputs,             \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        pg::MixedSensitivityOutputs, void*, std::size_t,                    \
        lspg::ExerciseReplayStrategy)

AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SwaptionSide::payer);
AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SwaptionSide::receiver);

#undef AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES

}  // namespace ai_factory::workbench::model::fixed_income::${binding_namespace}
