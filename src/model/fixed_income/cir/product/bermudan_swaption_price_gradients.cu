// Generated CIR Bermudan frozen-policy replay.
#include "model/fixed_income/cir/product/bermudan_swaption_price_gradients.cuh"

#include "common/fixed_income/bermudan_swaption_continuation_state.cuh"
#include "common/fixed_income/scalar_rate_continuation_state.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_launcher.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "common/simulation/early_exercise_schedule.cuh"
#include "common/simulation/terminal_forward_exercise_schedule.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "model/fixed_income/cir/forward_measure_impl.cuh"
#include "product/bermudan_swaption/price_gradients/device_prepared_frozen_exercise_node_graph.cuh"
#include "product/bermudan_swaption/price_gradients/device_prepared_pricing_policy.cuh"
#include "product/bermudan_swaption/pricing_policy.cuh"
#include "product/bermudan_swaption/terminal_forward_pricing_policy.cuh"

namespace ai_factory::workbench::model::fixed_income::cir {
namespace {

namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
namespace lspg =
    ::ai_factory::workbench::longstaff_schwartz::price_gradients;

using Dynamics = cir::terminal_forward::DynamicsPolicy;
using Schedule = simulation::TerminalForwardRegularExerciseSchedule<Dynamics>;
using Analytics = BermudanSwaptionAnalyticsPolicy;
using ContinuationState = ::ai_factory::workbench::fixed_income::ScalarRateContinuationState<Dynamics>;
template<SwaptionSide Side>
using CentralPolicy = product::TerminalForwardBermudanSwaptionPricingPolicy<
    product::StandaloneBermudanSwaptionPricingPolicy<
        Schedule, Analytics, Side, ContinuationState
    >,
    terminal_forward::BondAnalyticsPolicy
>;

template<SwaptionSide Side, pg::SensitivityOrders Orders>
using Policy = product::BermudanSwaptionDevicePreparedSensitivityPolicy<
    CentralPolicy<Side>,
    typename BermudanSwaptionPriceGradientPlan::Preparation,
    typename BermudanSwaptionPriceGradientPlan::DeviceInputs,
    Orders,
    true
>;
constexpr std::size_t kNodeGraphMaximumSensitivities = 7U;
constexpr unsigned int kNodeGraphGroupSize = 16U;
constexpr unsigned int kNodeGraphNodesPerWorker = 2U;
using NodeGraphTuning = mcpg::tuning::DefaultTerminalNodeTuning;
template<SwaptionSide Side, pg::SensitivityOrders Orders>
using NodeGraphPolicy = bermudan_pg::FrozenExerciseNodeGraphPolicy<
    Policy<Side, Orders>,
    kNodeGraphMaximumSensitivities,
    kNodeGraphGroupSize,
    kNodeGraphNodesPerWorker,
    NodeGraphTuning
>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::OneFactorHermiteBasis<3U>
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
        "cir.bermudan_swaption.price_gradients.prepare_stencils"
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
        "cir.bermudan_swaption.diagonal.prepare_stencils"
    );
}

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_cir_bermudan_swaption_price_gradients_cuda(
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
            "cir.bermudan_swaption.sensitivities",
            Side == SwaptionSide::payer
                ? "payer/nodes=3/B=1" : "receiver/nodes=3/B=1",
            "CIR Bermudan sensitivities"
        );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_cir_bermudan_swaption_diagonal_sensitivities_cuda(
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
            "cir.bermudan_swaption.sensitivities",
            Side == SwaptionSide::payer
                ? "payer/nodes=4/B=1" : "receiver/nodes=4/B=1",
            "CIR Bermudan sensitivities"
        );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
std::size_t cir_bermudan_swaption_node_graph_workspace_bytes(
    const BermudanSwaptionPriceGradientPlan& host,
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

template<SwaptionSide Side, pg::SensitivityOrders Orders>
longstaff_schwartz::LaunchResult
launch_cir_bermudan_swaption_node_graph_sensitivities_cuda(
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
        "cir.bermudan_swaption.sensitivities",
        Side == SwaptionSide::payer
            ? "payer/nodes=graph" : "receiver/nodes=graph",
        "CIR Bermudan node-graph sensitivities"
    );
}

#define AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SIDE)                 \
    template longstaff_schwartz::LaunchResult                               \
    launch_cir_bermudan_swaption_price_gradients_cuda<SIDE>( \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::StencilOutputs,                  \
        const pg::LaunchConfiguration&, pg::Outputs);                       \
    template longstaff_schwartz::LaunchResult                               \
    launch_cir_bermudan_swaption_diagonal_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs);            \
    template longstaff_schwartz::LaunchResult                               \
    launch_cir_bermudan_swaption_diagonal_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs);            \
    template std::size_t                                                    \
    cir_bermudan_swaption_node_graph_workspace_bytes<        \
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&);                                    \
    template std::size_t                                                    \
    cir_bermudan_swaption_node_graph_workspace_bytes<        \
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        const pg::LaunchConfiguration&);                                    \
    template longstaff_schwartz::LaunchResult                               \
    launch_cir_bermudan_swaption_node_graph_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::second>(                               \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        void*, std::size_t);                                                \
    template longstaff_schwartz::LaunchResult                               \
    launch_cir_bermudan_swaption_node_graph_sensitivities_cuda<\
        SIDE, pg::SensitivityOrders::first_and_second>(                     \
        const BermudanSwaptionPriceGradientPlan&,                           \
        BermudanSwaptionPriceGradientPlan::DeviceInputs,                    \
        BermudanSwaptionPriceGradientPlan::DiagonalStencilOutputs,          \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs,             \
        void*, std::size_t)

AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SwaptionSide::payer);
AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES(SwaptionSide::receiver);

#undef AI_FACTORY_INSTANTIATE_BERMUDAN_SENSITIVITIES

}  // namespace ai_factory::workbench::model::fixed_income::cir
