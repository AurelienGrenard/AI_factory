// Schobel-Zhu phoenix-autocall composition over generic CUDA layers.
#include "model/equity/markovian/schobel_zhu/product/phoenix_autocall.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"
#include "product/phoenix_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::schobel_zhu {
namespace {

using Schedule = simulation::FixedStepRegularSchedule<schobel_zhu::DynamicsPolicy>;
using PricingPolicy = product::PhoenixAutocallPricingPolicy<Schedule>;

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy>);

}  // namespace

void launch_schobel_zhu_phoenix_autocall_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::PhoenixAutocallParameters* host_products,
    const product::PhoenixAutocallParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
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
    monte_carlo::launch_monte_carlo_cuda<PricingPolicy>(
        make_model_product_device_inputs(
            device_models,
            model_count,
            device_products,
            product_count,
            construction
        ),
        {host_products, product_count, construction},
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
        "schobel_zhu.phoenix_autocall",
        "default",
        "Schobel-Zhu phoenix autocall kernel"
    );
}



}  // namespace ai_factory::workbench::model::equity::schobel_zhu
