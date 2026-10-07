// Generated OU joint-transition composition for a grid-monitored bond barrier.
#include "model/fixed_income/${model}/product/zero_coupon_bond_up_and_out.cuh"
#include "model/fixed_income/${model}/dynamics_impl.cuh"
#include "model/fixed_income/${model}/analytics_impl.cuh"
#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "product/zero_coupon_bond_up_and_out/pricing_policy.cuh"

namespace ai_factory::workbench::model::fixed_income::${model} {
namespace {
using Schedule = simulation::FixedStepDenseSchedule<joint::DynamicsPolicy>;
using Pricing = product::ZeroCouponBondUpAndOutPricingPolicy<Schedule>;
static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<Pricing>);
}  // namespace

void launch_${model}_zero_coupon_bond_up_and_out_cuda(
    const ModelParameters* device_models, std::size_t model_count,
    const product::ZeroCouponBondUpAndOutParameters* host_products,
    const product::ZeroCouponBondUpAndOutParameters* device_products,
    std::size_t product_count, PriceConstruction construction,
    std::size_t result_count, std::size_t result_offset,
    std::size_t launch_result_count, std::size_t paths_per_price,
    std::uint32_t simulation_steps_per_day, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t base_seed,
    float* device_prices, float* device_standard_errors
) {
    monte_carlo::launch_monte_carlo_cuda<Pricing>(
        make_model_product_device_inputs(device_models, model_count,
            device_products, product_count, construction),
        {host_products, product_count, construction},
        result_count, result_offset, launch_result_count, paths_per_price,
        simulation::FixedStepTimeConfiguration{
            1.0f / (252.0f * static_cast<float>(simulation_steps_per_day)),
            simulation_steps_per_day},
        threads_per_block, block_count, base_seed, device_prices,
        device_standard_errors,
        "${model}.zero_coupon_bond_up_and_out", "call",
        "${model_display} zero-coupon bond barrier Monte Carlo kernel"
    );
}

}  // namespace ai_factory::workbench::model::fixed_income::${model}
