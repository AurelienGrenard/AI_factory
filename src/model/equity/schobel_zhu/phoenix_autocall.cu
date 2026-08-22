// Schobel-Zhu phoenix-autocall composition over generic CUDA layers.
#include "model/equity/schobel_zhu/phoenix_autocall.cuh"

#include "common/equity/discount.cuh"
#include "common/equity/monte_carlo_kernel.cuh"
#include "common/equity/schedule.cuh"
#include "model/equity/schobel_zhu/dynamics.cu"
#include "product/phoenix_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::schobel_zhu {
namespace {

using Schedule = equity::FixedStepRegularSchedule<schobel_zhu::DynamicsPolicy>;
using Discount = equity::ConstantRateDiscountPolicy<schobel_zhu::DynamicsPolicy>;
using PricingPolicy = product::PhoenixAutocallPricingPolicy<Schedule, Discount>;

static_assert(equity::EquitySchedulePolicy<Schedule>);
static_assert(equity::ScalarMonteCarloPricingPolicy<PricingPolicy>);

}  // namespace

void launch_schobel_zhu_phoenix_autocall_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::PhoenixAutocallParameters* device_products,
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
        "schobel_zhu.phoenix_autocall",
        "default",
        "Schobel-Zhu phoenix autocall kernel"
    );
}

}  // namespace ai_factory::workbench::schobel_zhu
