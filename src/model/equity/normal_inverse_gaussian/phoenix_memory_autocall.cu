// Normal-Inverse-Gaussian phoenix-memory-autocall composition over generic CUDA layers.
#include "model/equity/normal_inverse_gaussian/phoenix_memory_autocall.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/normal_inverse_gaussian/dynamics.cu"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {
namespace {

using Schedule = simulation::ExactTransitionRegularSchedule<normal_inverse_gaussian::DynamicsPolicy>;
using PricingPolicy = product::PhoenixMemoryAutocallPricingPolicy<Schedule>;

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy>);

}  // namespace

void launch_normal_inverse_gaussian_phoenix_memory_autocall_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::PhoenixMemoryAutocallParameters* device_products,
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
    monte_carlo::launch_monte_carlo_cuda<PricingPolicy>(
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
        "normal_inverse_gaussian.phoenix_memory_autocall",
        "default",
        "Normal-Inverse-Gaussian phoenix memory autocall kernel"
    );
}

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
