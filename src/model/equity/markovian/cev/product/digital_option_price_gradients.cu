// Generated cev fixed-step digital_option sensitivities over the common device-prepared engine.
#include "model/equity/markovian/cev/product/digital_option_price_gradients.cuh"

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/terminal_product_sensitivity_policy.cuh"
#include "model/equity/markovian/cev/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/cev/product/digital_option.cuh"
#include "product/digital_option/pricing_policy.cuh"

#include <algorithm>

namespace ai_factory::workbench::model::equity::cev {

void prepare_digital_option_price_gradient_stencils_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_first_sensitivity_stencils(
        host,
        device,
        stencil_outputs,
        result_offset,
        result_count,
        "cev.digital_option.price_gradients.stencil_preparation"
    );
}

void prepare_digital_option_diagonal_sensitivity_stencils_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    epg::prepare_terminal_sensitivity_stencils<
        pg::SensitivityOrders::first_and_second
    >(
        host, device, stencil_outputs, result_offset, result_count,
        "cev.digital_option.diagonal.stencil_preparation"
    );
}

template<OptionSide Side>
void launch_cev_digital_option_price_gradients_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    const auto launch_price_only = [&] {
        launch_cev_digital_option_cuda<Side>(
            device.models,
            host.models.size(),
            host.products.data(),
            device.products,
            host.products.size(),
            host.construction,
            host.result_count,
            configuration.result_offset,
            configuration.result_count,
            configuration.paths_per_price,
            host.time.dt,
            host.time.simulation_steps_per_day,
            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_first_sensitivities<
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "cev.digital_option.price_gradients.device_prepared",
        Side == OptionSide::call ? "call/nodes=3/B=1" : "put/nodes=3/B=1"
    );
}

template<OptionSide Side, pg::SensitivityOrders Orders>
void launch_cev_digital_option_diagonal_sensitivities_cuda(
    const DigitalOptionPriceGradientPlan& host,
    DigitalOptionPriceGradientPlan::DeviceInputs device,
    DigitalOptionPriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs
) {
    const auto launch_price_only = [&] {
        launch_cev_digital_option_cuda<Side>(
            device.models,
            host.models.size(),
            host.products.data(),
            device.products,
            host.products.size(),
            host.construction,
            host.result_count,
            configuration.result_offset,
            configuration.result_count,
            configuration.paths_per_price,
            host.time.dt,
            host.time.simulation_steps_per_day,
            configuration.threads_per_block,
            std::min(configuration.result_count, configuration.block_count),
            configuration.base_seed,
            outputs.prices,
            outputs.price_standard_errors
        );
    };
    epg::launch_terminal_diagonal_sensitivities<
        Orders,
        mpg::CoupledDynamics,
        epg::TerminalProductSensitivityPolicy<product::DigitalOptionPathPolicy<Side>>
    >(
        host,
        device,
        stencil_outputs,
        configuration,
        outputs,
        launch_price_only,
        "cev.digital_option.sensitivities.device_prepared",
        Orders == pg::SensitivityOrders::second
            ? "diagonal_hessian/nodes=4/B=1"
            : "gradient_and_diagonal_hessian/nodes=4/B=1"
    );
}

#define AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES(SIDE) \
    template void launch_cev_digital_option_price_gradients_cuda<SIDE>( \
        const DigitalOptionPriceGradientPlan&, \
        DigitalOptionPriceGradientPlan::DeviceInputs, \
        DigitalOptionPriceGradientPlan::StencilOutputs, \
        const pg::LaunchConfiguration&, pg::Outputs); \
    template void launch_cev_digital_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::second>( \
        const DigitalOptionPriceGradientPlan&, \
        DigitalOptionPriceGradientPlan::DeviceInputs, \
        DigitalOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs); \
    template void launch_cev_digital_option_diagonal_sensitivities_cuda< \
        SIDE, pg::SensitivityOrders::first_and_second>( \
        const DigitalOptionPriceGradientPlan&, \
        DigitalOptionPriceGradientPlan::DeviceInputs, \
        DigitalOptionPriceGradientPlan::DiagonalStencilOutputs, \
        const pg::LaunchConfiguration&, pg::SensitivityOutputs)

AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES(OptionSide::call);
AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES(OptionSide::put);

#undef AI_FACTORY_INSTANTIATE_TERMINAL_SENSITIVITIES

}  // namespace ai_factory::workbench::model::equity::cev
