// Quadratic rough-Heston athena-autocall composition over a prepared N-factor lift.
#include "model/equity/rough/quadratic_rough_heston/product/athena_autocall.cuh"

#include "common/simulation/schedule.cuh"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_pricing.cuh"
#include "product/athena_autocall/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {
namespace {

template<std::size_t FactorCount>
using Schedule = simulation::FixedStepRegularSchedule<DynamicsPolicy<FactorCount>>;

}  // namespace

template<std::size_t FactorCount>
void launch_quadratic_rough_heston_athena_autocall_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::AthenaAutocallParameters* host_products,
    const product::AthenaAutocallParameters* device_products,
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
    using ProductPolicy = product::AthenaAutocallPathPolicy;
    launch_quadratic_rough_heston_markovian_cuda<
        FactorCount,
        ProductPolicy,
        Schedule<FactorCount>
    >(
        device_models,
        model_count,
        device_prepared_dynamics,
        prepared_dynamics_count,
        host_products,
        device_products,
        product_count,
        construction,
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        dt,
        simulation_steps_per_day,
        threads_per_block,
        block_count,
        base_seed,
        device_prices,
        device_standard_errors,
        "quadratic_rough_heston.athena_autocall",
        "default",
        "Quadratic rough-Heston athena autocall kernel"
    );
}

template void launch_quadratic_rough_heston_athena_autocall_cuda<
    2U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<2U>*, std::size_t,
    const product::AthenaAutocallParameters*,
    const product::AthenaAutocallParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

template void launch_quadratic_rough_heston_athena_autocall_cuda<
    3U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<3U>*, std::size_t,
    const product::AthenaAutocallParameters*,
    const product::AthenaAutocallParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

template void launch_quadratic_rough_heston_athena_autocall_cuda<
    7U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<7U>*, std::size_t,
    const product::AthenaAutocallParameters*,
    const product::AthenaAutocallParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
