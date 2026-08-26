// Merton digital-option composition over generic CUDA layers.
#include "model/equity/merton/digital_option.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/merton/dynamics.cu"
#include "product/digital_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::merton {
namespace {

using Schedule = simulation::ExactTransitionTerminalSchedule<merton::DynamicsPolicy>;
template<OptionSide Side>
using PricingPolicy =
    product::DigitalOptionPricingPolicy<Schedule, Side>;

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy<OptionSide::call>>);

}  // namespace

template<OptionSide Side>
void launch_merton_digital_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::DigitalOptionParameters* device_products,
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
    const simulation::ExactTransitionTimeConfiguration time_configuration{
        day_fraction,
    };
    monte_carlo::launch_monte_carlo_cuda<PricingPolicy<Side>>(
        make_model_product_device_inputs(
            device_models,
            model_count,
            device_products,
            product_count,
            cartesian_product
        ),
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        time_configuration,
        threads_per_block,
        block_count,
        base_seed,
        device_prices,
        device_standard_errors,
        "merton.digital_option",
        option_side_name(Side),
        "Merton digital option kernel"
    );
}

template void launch_merton_digital_option_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::DigitalOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
    float, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);
template void launch_merton_digital_option_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::DigitalOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
    float, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

}  // namespace ai_factory::workbench::model::equity::merton
