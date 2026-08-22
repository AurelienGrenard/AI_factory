// Experimental Heston Asian-option composition over generic CUDA layers.
#include "model/equity/heston/asian_optionbis.cuh"

#include "common/equity/discount.cuh"
#include "common/equity/monte_carlo.cuh"
#include "common/equity/schedule.cuh"
#include "model/equity/heston/dynamicsbis.cu"
#include "product/asian_option/pricing_policybis.cuh"

namespace ai_factory::workbench::heston {
namespace {

using Schedule = equity::FixedStepDenseSchedule<heston::DynamicsPolicy>;
using Discount = equity::ConstantRateDiscountPolicy<heston::DynamicsPolicy>;

template<OptionSide Side>
using PricingPolicy =
    product::AsianOptionPricingPolicy<Schedule, Discount, Side>;

static_assert(equity::EquitySchedulePolicy<Schedule>);
static_assert(equity::DiscountPolicyFor<Discount, heston::DynamicsPolicy>);
static_assert(
    equity::ScalarMonteCarloPricingPolicy<PricingPolicy<OptionSide::call>>
);

}  // namespace

template<OptionSide Side>
void launch_heston_asian_optionbis_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::AsianOptionParameters* device_products,
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
    const typename PricingPolicy<Side>::DeviceInputs inputs{};
    equity::launch_monte_carlo_cuda<PricingPolicy<Side>>(
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
        "heston.asian_optionbis",
        option_side_name(Side),
        "Heston Asian option bis kernel"
    );
}

template void launch_heston_asian_optionbis_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::AsianOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);
template void launch_heston_asian_optionbis_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::AsianOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

}  // namespace ai_factory::workbench::heston
