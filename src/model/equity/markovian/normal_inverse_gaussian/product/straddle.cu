// Normal-Inverse-Gaussian straddle composition over generic CUDA layers.
#include "model/equity/markovian/normal_inverse_gaussian/product/straddle.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dynamics_impl.cuh"
#include "product/straddle/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {
namespace {

using Schedule = simulation::ExactTransitionTerminalSchedule<normal_inverse_gaussian::DynamicsPolicy>;
using PricingPolicy = product::StraddlePricingPolicy<Schedule>;

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy>);

}  // namespace

void launch_normal_inverse_gaussian_straddle_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::StraddleParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
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
            construction
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
        "normal_inverse_gaussian.straddle",
        "default",
        "Normal-Inverse-Gaussian straddle kernel"
    );
}



}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
