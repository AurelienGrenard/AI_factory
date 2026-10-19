// CIR++/Flat bond-barrier first-order CRN sensitivities.
#include "model/fixed_income/cir_plus_plus/product/flat/zero_coupon_bond_up_and_out_price_gradients.cuh"

#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/fixed_income/price_gradients/bond_barrier_first_policy.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_first_sensitivity_kernel.cuh"
#include "common/price_gradients/device_prepared_stencil_launcher.cuh"
#include "product/zero_coupon_bond_up_and_out/pricing_policy.cuh"
#include "model/fixed_income/cir_plus_plus/dynamics.cuh"
#include "model/fixed_income/cir/dynamics_impl.cuh"
#include "model/fixed_income/cir_plus_plus/flat/analytics_impl.cuh"
#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::flat {

namespace epg = ::ai_factory::workbench::equity::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

namespace {
using Schedule = simulation::FixedStepDenseSchedule<
    cir_plus_plus::joint::DynamicsPolicy>;
using Pricing = product::FittedZeroCouponBondUpAndOutPricingPolicy<Schedule, FittedModelComposition>;
using BarrierFirstPolicy = fipg::BondBarrierFirstPolicy<
    Pricing, ZeroCouponBondUpAndOutPriceGradientPlan::Preparation>;
}

void prepare_zero_coupon_bond_up_and_out_price_gradient_stencils_cuda(
    const ZeroCouponBondUpAndOutPriceGradientPlan& host,
    ZeroCouponBondUpAndOutPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondUpAndOutPriceGradientPlan::StencilOutputs stencil_outputs,
    std::size_t result_offset,
    std::size_t result_count
) {
    pg::prepare_device_sensitivity_stencils<pg::SensitivityOrders::first>(
        host, device, stencil_outputs, result_offset, result_count,
        "cir_plus_plus.flat.zero_coupon_bond_up_and_out.sensitivities.prepare_stencils"
    );
}

void launch_cir_plus_plus_flat_zero_coupon_bond_up_and_out_price_gradients_cuda(
    const ZeroCouponBondUpAndOutPriceGradientPlan& host,
    ZeroCouponBondUpAndOutPriceGradientPlan::DeviceInputs device,
    ZeroCouponBondUpAndOutPriceGradientPlan::StencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs
) {
    if (configuration.method != pg::PricingMethod::monte_carlo) {
        throw std::invalid_argument("Bond-barrier gradients require Monte Carlo.");
    }
    if (std::fabs(static_cast<double>(host.time.dt)
            * host.time.simulation_steps_per_day - 1.0 / 252.0) > 1e-8) {
        throw std::invalid_argument("Bond-barrier monitoring requires a 252-day calendar.");
    }
    epg::validate_terminal_first_sensitivity_launch(
        host, device, stencil_outputs, configuration, outputs
    );
    mcpg::launch_device_prepared_first_sensitivities<
        BarrierFirstPolicy,
        ZeroCouponBondUpAndOutPriceGradientPlan::Preparation
    >(
        device,
        mcpg::make_device_prepared_plan(host),
        configuration,
        outputs,
        stencil_outputs,
        "cir_plus_plus.flat.zero_coupon_bond_up_and_out.sensitivities.device_prepared",
        "call/grid_barrier/crn"
    );
}

}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus::flat
