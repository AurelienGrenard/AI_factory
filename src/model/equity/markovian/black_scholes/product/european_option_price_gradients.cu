// Generated Black-Scholes analytical and MC sensitivities with device-prepared rows.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"

#include "common/closed_form/price_gradients/device_prepared_kernel.cuh"
#include "common/closed_form/price_gradients/device_prepared_mixed_kernel.cuh"
#include "common/equity/price_gradients/mixed_terminal_node_graph_launcher.cuh"
#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_terminal_price_kernel.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"
#include "model/equity/markovian/black_scholes/price_gradients/coupled_dynamics_impl.cuh"
#include "product/european_option/price_gradients/closed_form_policy.cuh"
#include "product/european_option/price_gradients/monte_carlo_policy.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

void prepare_european_option_price_gradient_stencils_cuda(
    const EuropeanOptionPriceGradientPlan& host,
    EuropeanOptionPriceGradientPlan::DeviceInputs device,
    EuropeanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "black_scholes.european_option.price_gradients.stencil_preparation"
    );
}

void prepare_european_option_diagonal_sensitivity_stencils_cuda(
    const EuropeanOptionPriceGradientPlan& host,
    EuropeanOptionPriceGradientPlan::DeviceInputs device,
    EuropeanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "black_scholes.european_option.diagonal.stencil_preparation"
    );
}

template<OptionSide Side>
void launch_black_scholes_european_option_price_gradients_cuda(
    const EuropeanOptionPriceGradientPlan& host,
    EuropeanOptionPriceGradientPlan::DeviceInputs device,
    EuropeanOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    using AnalyticalPolicy =
        product::european_option::price_gradients::ClosedFormPolicy<
            ModelParameters,
            Side
        >;
    if (configuration.method == pg::PricingMethod::closed_form) {
        const pg::SensitivityOutputs analytical_outputs{
            outputs.prices,
            nullptr,
            outputs.gradients,
            nullptr,
            nullptr,
            nullptr,
            outputs.price_capacity,
            outputs.gradient_capacity,
        };
        closed_form::price_gradients::launch_device_prepared<
            pg::SensitivityOrders::first,
            AnalyticalPolicy
        >(
            host,
            device,
            stencil_outputs,
            configuration,
            analytical_outputs,
            "black_scholes.european_option.price_gradients.closed_form",
            Side == OptionSide::call ? "call/nodes=3" : "put/nodes=3"
        );
        return;
    }
    const auto launch_price_only = [&] {
        mcpg::launch_device_prepared_terminal_prices<
            mpg::CoupledDynamics,
            product::EuropeanOptionGradientPathPolicy<Side>
        >(
            host,
            device,
            stencil_outputs,
            configuration,
            outputs,
            "black_scholes.european_option.price_gradients.price_only",
            Side == OptionSide::call ? "call/B=1" : "put/B=1"
        );
    };
    epg::launch_terminal_first_sensitivities<
        mpg::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "black_scholes.european_option.price_gradients.monte_carlo",
        Side == OptionSide::call ? "call/nodes=3/B=1" : "put/nodes=3/B=1"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_black_scholes_european_option_diagonal_sensitivities_cuda(
    const EuropeanOptionPriceGradientPlan& host,
    EuropeanOptionPriceGradientPlan::DeviceInputs device,
    EuropeanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    using AnalyticalPolicy =
        product::european_option::price_gradients::ClosedFormPolicy<
            ModelParameters,
            Side
        >;
    if (configuration.method == pg::PricingMethod::closed_form) {
        closed_form::price_gradients::launch_device_prepared<
            Orders,
            AnalyticalPolicy
        >(
            host,
            device,
            stencil_outputs,
            configuration,
            outputs,
            "black_scholes.european_option.sensitivities.closed_form",
            Orders == pg::SensitivityOrders::second
                ? "diagonal_hessian/nodes=4"
                : "gradient_and_diagonal_hessian/nodes=4"
        );
        return;
    }
    const auto launch_price_only = [&] {
        const typename EuropeanOptionPriceGradientPlan::StencilOutputs
            first_stencil_outputs{
                nullptr,
                0U,
                stencil_outputs.error,
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
            product::EuropeanOptionGradientPathPolicy<Side>
        >(
            host,
            device,
            first_stencil_outputs,
            configuration,
            price_outputs,
            "black_scholes.european_option.sensitivities.price_only",
            Side == OptionSide::call ? "call/B=1" : "put/B=1"
        );
    };
    epg::launch_terminal_diagonal_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "black_scholes.european_option.sensitivities.monte_carlo",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

template<OptionSide Side>
std::size_t black_scholes_european_option_mixed_node_graph_workspace_bytes(
    const EuropeanOptionPriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
) {
    if (configuration.method == pg::PricingMethod::closed_form) {
        return closed_form::price_gradients::mixed_workspace_bytes(host);
    }
    return epg::mixed_terminal_node_graph_workspace_bytes<
        mpg::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>,
        6U,
        15U,
        64U,
        2U
    >(host, configuration);
}

template<OptionSide Side>
void launch_black_scholes_european_option_mixed_node_graph_sensitivities_cuda(
    const EuropeanOptionPriceGradientPlan& host,
    EuropeanOptionPriceGradientPlan::DeviceInputs device,
    EuropeanOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    EuropeanOptionPriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
) {
    if (configuration.method == pg::PricingMethod::closed_form) {
        using AnalyticalPolicy =
            product::european_option::price_gradients::ClosedFormPolicy<
                ModelParameters,
                Side
            >;
        closed_form::price_gradients::launch_device_prepared_mixed<
            AnalyticalPolicy,
            6U,
            15U
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
            "black_scholes.european_option.sensitivities.closed_form_mixed",
            Side == OptionSide::call ? "call" : "put"
        );
        return;
    }
    epg::launch_mixed_terminal_node_graph_sensitivities<
        mpg::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>,
        6U,
        15U,
        64U,
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
        "black_scholes.european_option.sensitivities.mixed_node_graph",
        "selected_gradient_and_hessian/nodes=graph"
    );
}

#define AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES(SIDE) \
    template void launch_black_scholes_european_option_price_gradients_cuda<SIDE>( \
        const EuropeanOptionPriceGradientPlan&, \
        EuropeanOptionPriceGradientPlan::DeviceInputs, \
        EuropeanOptionPriceGradientPlan::StencilOutputs, \
        const pg::LaunchConfiguration&, pg::Outputs); \
    template void launch_black_scholes_european_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const EuropeanOptionPriceGradientPlan&, \
        EuropeanOptionPriceGradientPlan::DeviceInputs, \
        EuropeanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template void launch_black_scholes_european_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const EuropeanOptionPriceGradientPlan&, \
        EuropeanOptionPriceGradientPlan::DeviceInputs, \
        EuropeanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template std::size_t \
    black_scholes_european_option_mixed_node_graph_workspace_bytes<SIDE>( \
        const EuropeanOptionPriceGradientPlan&, \
        const pg::LaunchConfiguration&); \
    template void \
    launch_black_scholes_european_option_mixed_node_graph_sensitivities_cuda<SIDE>( \
        const EuropeanOptionPriceGradientPlan&, \
        EuropeanOptionPriceGradientPlan::DeviceInputs, \
        EuropeanOptionPriceGradientPlan::DiagonalStencilOutputs, \
        EuropeanOptionPriceGradientPlan::MixedStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs, \
        pg::MixedSensitivityOutputs, void*, std::size_t)

AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES(OptionSide::call);
AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES(OptionSide::put);

#undef AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES

}  // namespace ai_factory::workbench::model::equity::black_scholes
