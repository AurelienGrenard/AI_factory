// G2++/Flat European-swaption mono/node-graph MC sensitivities.
#include "model/fixed_income/g2_plus_plus/product/flat/european_swaption_price_gradients.cuh"

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/mixed_terminal_node_graph_launcher.cuh"
#include "common/equity/price_gradients/terminal_node_graph_launcher.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_terminal_price_kernel.cuh"
#include "product/european_swaption/price_gradients/monte_carlo_policy.cuh"
#include "model/fixed_income/g2_plus_plus/price_gradients/coupled_dynamics_impl.cuh"
#include "model/fixed_income/g2_plus_plus/flat/analytics_impl.cuh"
#include <algorithm>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::flat {

namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
namespace mpg =
    ::ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients;

void prepare_european_swaption_price_gradient_stencils_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host, device, stencil_outputs, result_offset, result_count,
        "g2_plus_plus.flat.european_swaption.price_gradients.stencil_preparation"
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
        "g2_plus_plus.flat.european_swaption.diagonal.stencil_preparation"
    );
}

template<SwaptionSide Side>
void launch_g2_plus_plus_flat_european_swaption_price_gradients_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    using ProductPolicy = european_swaption_pg::FittedMonteCarloPolicy<
        FittedModelComposition, Side
    >;
    const auto launch_price_only = [&] {
        mcpg::launch_device_prepared_terminal_prices<
            mpg::CoupledDynamics,
            ProductPolicy
        >(
            host,
            device,
            stencil_outputs,
            configuration,
            outputs,
            "g2_plus_plus.flat.european_swaption.price_gradients.price_only",
            swaption_side_name(Side)
        );
    };
    epg::launch_terminal_first_sensitivities<
        mpg::CoupledDynamics,
        ProductPolicy
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "g2_plus_plus.flat.european_swaption.price_gradients.device_prepared",
        swaption_side_name(Side)
    );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
void launch_g2_plus_plus_flat_european_swaption_diagonal_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    using ProductPolicy = european_swaption_pg::FittedMonteCarloPolicy<
        FittedModelComposition, Side
    >;
    const auto launch_price_only = [&] {
        const typename EuropeanSwaptionPriceGradientPlan::StencilOutputs empty{
            nullptr, 0U, stencil_outputs.error
        };
        const pg::Outputs price_outputs{
            outputs.prices,
            outputs.price_standard_errors,
            nullptr,
            nullptr,
            outputs.price_capacity,
            0U,
        };
        mcpg::launch_device_prepared_terminal_prices<
            mpg::CoupledDynamics,
            ProductPolicy
        >(
            host,
            device,
            empty,
            configuration,
            price_outputs,
            "g2_plus_plus.flat.european_swaption.sensitivities.price_only",
            swaption_side_name(Side)
        );
    };
    epg::launch_terminal_diagonal_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        ProductPolicy
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "g2_plus_plus.flat.european_swaption.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
std::size_t g2_plus_plus_flat_european_swaption_node_graph_workspace_bytes(
    const EuropeanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    using ProductPolicy = european_swaption_pg::FittedMonteCarloPolicy<
        FittedModelComposition, Side
    >;
    return epg::terminal_node_graph_workspace_bytes<
        Orders,
        mpg::CoupledDynamics,
        ProductPolicy,
        10U,
        16U,
        2U
    >(host, configuration);
}

template<SwaptionSide Side, pg::SensitivityOrders Orders>
void launch_g2_plus_plus_flat_european_swaption_node_graph_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    using ProductPolicy = european_swaption_pg::FittedMonteCarloPolicy<
        FittedModelComposition, Side
    >;
    const auto launch_price_only = [&] {
        const typename EuropeanSwaptionPriceGradientPlan::StencilOutputs empty{
            nullptr, 0U, stencil_outputs.error
        };
        const pg::Outputs price_outputs{
            outputs.prices,
            outputs.price_standard_errors,
            nullptr,
            nullptr,
            outputs.price_capacity,
            0U,
        };
        mcpg::launch_device_prepared_terminal_prices<
            mpg::CoupledDynamics,
            ProductPolicy
        >(
            host,
            device,
            empty,
            configuration,
            price_outputs,
            "g2_plus_plus.flat.european_swaption.sensitivities.price_only",
            swaption_side_name(Side)
        );
    };
    epg::launch_terminal_node_graph_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        ProductPolicy,
        10U,
        16U,
        2U
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        workspace,
        workspace_bytes,
        launch_price_only,
        "g2_plus_plus.flat.european_swaption.sensitivities.node_graph",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=graph"
            : "gradient_and_diagonal_hessian/nodes=graph"
    );
}

template<SwaptionSide Side>
std::size_t g2_plus_plus_flat_european_swaption_mixed_node_graph_workspace_bytes(
    const EuropeanSwaptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    using ProductPolicy = european_swaption_pg::FittedMonteCarloPolicy<
        FittedModelComposition, Side
    >;
    return epg::mixed_terminal_node_graph_workspace_bytes<
        mpg::CoupledDynamics,
        ProductPolicy,
        10U,
        45U,
        128U,
        2U
    >(host, configuration);
}

template<SwaptionSide Side>
void launch_g2_plus_plus_flat_european_swaption_mixed_node_graph_sensitivities_cuda(
    const EuropeanSwaptionPriceGradientPlan& host,
    EuropeanSwaptionPriceGradientPlan::DeviceInputs device,
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    EuropeanSwaptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    using ProductPolicy = european_swaption_pg::FittedMonteCarloPolicy<
        FittedModelComposition, Side
    >;
    epg::launch_mixed_terminal_node_graph_sensitivities<
        mpg::CoupledDynamics,
        ProductPolicy,
        10U,
        45U,
        128U,
        2U
    >(
        host,
        device,
        stencil_outputs,
        mixed_stencil_outputs,
        configuration,
        outputs,
        mixed_outputs,
        workspace,
        workspace_bytes,
        "g2_plus_plus.flat.european_swaption.sensitivities.mixed_node_graph",
        "selected_gradient_and_hessian/nodes=graph"
    );
}

#define AI_FACTORY_INSTANTIATE(side, orders)                                \
template void launch_g2_plus_plus_flat_european_swaption_diagonal_sensitivities_cuda< \
    side, orders>(const EuropeanSwaptionPriceGradientPlan&,                 \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs,              \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs);                \
template std::size_t g2_plus_plus_flat_european_swaption_node_graph_workspace_bytes< \
    side, orders>(const EuropeanSwaptionPriceGradientPlan&,                 \
    const pg::LaunchConfiguration&);                                        \
template void launch_g2_plus_plus_flat_european_swaption_node_graph_sensitivities_cuda< \
    side, orders>(const EuropeanSwaptionPriceGradientPlan&,                 \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs,              \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs, void*, std::size_t)

#define AI_FACTORY_INSTANTIATE_SIDE(side)                                   \
template void launch_g2_plus_plus_flat_european_swaption_price_gradients_cuda< \
    side>(const EuropeanSwaptionPriceGradientPlan&,                         \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::StencilOutputs,                      \
    const pg::LaunchConfiguration&, pg::Outputs);                           \
template std::size_t                                                        \
g2_plus_plus_flat_european_swaption_mixed_node_graph_workspace_bytes<side>( \
    const EuropeanSwaptionPriceGradientPlan&,                               \
    const pg::LaunchConfiguration&);                                        \
template void                                                               \
launch_g2_plus_plus_flat_european_swaption_mixed_node_graph_sensitivities_cuda<side>( \
    const EuropeanSwaptionPriceGradientPlan&,                               \
    EuropeanSwaptionPriceGradientPlan::DeviceInputs,                        \
    EuropeanSwaptionPriceGradientPlan::DiagonalStencilOutputs,              \
    EuropeanSwaptionPriceGradientPlan::MixedStencilOutputs,                 \
    const pg::LaunchConfiguration&, pg::SensitivityOutputs,                 \
    pg::MixedSensitivityOutputs, void*, std::size_t);                       \
AI_FACTORY_INSTANTIATE(side, pg::SensitivityOrders::second);                \
AI_FACTORY_INSTANTIATE(side, pg::SensitivityOrders::first_and_second)

AI_FACTORY_INSTANTIATE_SIDE(SwaptionSide::payer);
AI_FACTORY_INSTANTIATE_SIDE(SwaptionSide::receiver);
#undef AI_FACTORY_INSTANTIATE_SIDE
#undef AI_FACTORY_INSTANTIATE

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::flat
