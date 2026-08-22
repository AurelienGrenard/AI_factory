// Merton up-one-touch composition over generic CUDA layers.
#include "model/equity/merton/up_one_touch.cuh"

#include "common/equity/discount.cuh"
#include "common/equity/monte_carlo_kernel.cuh"
#include "common/equity/schedule.cuh"
#include "model/equity/merton/dynamics.cu"
#include "product/up_one_touch/pricing_policy.cuh"

namespace ai_factory::workbench::merton {
namespace {

using Schedule = equity::FixedStepDenseSchedule<merton::DynamicsPolicy>;
using Discount = equity::ConstantRateDiscountPolicy<merton::DynamicsPolicy>;
using PricingPolicy = product::UpOneTouchPricingPolicy<Schedule, Discount>;

static_assert(equity::EquitySchedulePolicy<Schedule>);
static_assert(equity::ScalarMonteCarloPricingPolicy<PricingPolicy>);

}  // namespace

void launch_merton_up_one_touch_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::UpOneTouchParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    const equity::FixedStepConfiguration configuration{
        dt,
        simulation_steps_per_day,
    };
    const typename PricingPolicy::DeviceInputs inputs{};
    equity::launch_monte_carlo_cuda<PricingPolicy>(
        device_models,
        model_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        configuration,
        inputs,
        threads_per_block,
        block_count,
        base_seed,
        device_prices,
        device_standard_errors,
        "merton.up_one_touch",
        "default",
        "Merton up one touch kernel"
    );
}

}  // namespace ai_factory::workbench::merton
