// CEV athena-autocall composition over generic CUDA layers.
#include "model/equity/cev/athena_autocall.cuh"

#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/cev/dynamics.cu"
#include "product/athena_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::cev {
namespace {

using Schedule = simulation::FixedStepRegularSchedule<cev::DynamicsPolicy>;
using PricingPolicy = product::AthenaAutocallPricingPolicy<Schedule>;

static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PricingPolicy>);

}  // namespace

void launch_cev_athena_autocall_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::AthenaAutocallParameters* device_products,
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
        "cev.athena_autocall",
        "default",
        "CEV athena autocall kernel"
    );
}

}  // namespace ai_factory::workbench::model::equity::cev
