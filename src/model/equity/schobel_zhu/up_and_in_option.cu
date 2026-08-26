// Schobel-Zhu up-and-in-option composition over generic CUDA layers.
#include "model/equity/schobel_zhu/up_and_in_option.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/schobel_zhu/dynamics.cu"
#include "product/up_and_in_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::schobel_zhu {
namespace {

using Schedule = simulation::FixedStepDenseSchedule<schobel_zhu::DynamicsPolicy>;
template<OptionSide Side>
using PricingPolicy =
    product::UpAndInOptionPricingPolicy<Schedule, Side>;

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy<OptionSide::call>>);

}  // namespace

template<OptionSide Side>
void launch_schobel_zhu_up_and_in_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::UpAndInOptionParameters* device_products,
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
    const simulation::FixedStepTimeConfiguration time_configuration{
        dt,
        simulation_steps_per_day,
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
        "schobel_zhu.up_and_in_option",
        option_side_name(Side),
        "Schobel-Zhu up and in option kernel"
    );
}

template void launch_schobel_zhu_up_and_in_option_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::UpAndInOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);
template void launch_schobel_zhu_up_and_in_option_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::UpAndInOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
