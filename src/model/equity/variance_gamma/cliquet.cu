// Variance-Gamma cliquet composition over generic CUDA layers.
#include "model/equity/variance_gamma/cliquet.cuh"

#include "common/equity/discount.cuh"
#include "common/equity/monte_carlo_kernel.cuh"
#include "common/equity/schedule.cuh"
#include "model/equity/variance_gamma/dynamics.cu"
#include "product/cliquet/pricing_policy.cuh"

namespace ai_factory::workbench::variance_gamma {
namespace {

using Schedule = equity::ExactTransitionRegularSchedule<variance_gamma::DynamicsPolicy>;
using Discount = equity::ConstantRateDiscountPolicy<variance_gamma::DynamicsPolicy>;
using PricingPolicy = product::CliquetPricingPolicy<Schedule, Discount>;

static_assert(equity::EquitySchedulePolicy<Schedule>);
static_assert(equity::ScalarMonteCarloPricingPolicy<PricingPolicy>);

}  // namespace

void launch_variance_gamma_cliquet_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::CliquetParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    const equity::ExactTransitionConfiguration configuration{
        day_fraction,
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
        "variance_gamma.cliquet",
        "default",
        "Variance-Gamma cliquet kernel"
    );
}

}  // namespace ai_factory::workbench::variance_gamma
