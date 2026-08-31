// Rough-Heston down-and-out-option composition over a prepared N-factor lift.
#include "model/equity/rough/rough_heston/product/down_and_out_option.cuh"

#include "common/simulation/schedule.cuh"
#include "model/equity/rough/rough_heston/markovian_n_factor_pricing.cuh"
#include "product/down_and_out_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::rough_heston {
namespace {

template<std::size_t FactorCount>
using Schedule = simulation::FixedStepDenseSchedule<DynamicsPolicy<FactorCount>>;

}  // namespace

template<OptionSide Side, std::size_t FactorCount>
void launch_rough_heston_down_and_out_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::DownAndOutOptionParameters* host_products,
    const product::DownAndOutOptionParameters* device_products,
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
    using ProductPolicy = product::DownAndOutOptionPathPolicy<Side>;
    launch_rough_heston_markovian_cuda<
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
        "rough_heston.down_and_out_option",
        option_side_name(Side),
        "Rough-Heston down and out option kernel"
    );
}

template void launch_rough_heston_down_and_out_option_cuda<
    OptionSide::call, 2U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<2U>*, std::size_t,
    const product::DownAndOutOptionParameters*,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

template void launch_rough_heston_down_and_out_option_cuda<
    OptionSide::put, 2U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<2U>*, std::size_t,
    const product::DownAndOutOptionParameters*,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

template void launch_rough_heston_down_and_out_option_cuda<
    OptionSide::call, 3U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<3U>*, std::size_t,
    const product::DownAndOutOptionParameters*,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

template void launch_rough_heston_down_and_out_option_cuda<
    OptionSide::put, 3U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<3U>*, std::size_t,
    const product::DownAndOutOptionParameters*,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

template void launch_rough_heston_down_and_out_option_cuda<
    OptionSide::call, 7U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<7U>*, std::size_t,
    const product::DownAndOutOptionParameters*,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

template void launch_rough_heston_down_and_out_option_cuda<
    OptionSide::put, 7U
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<7U>*, std::size_t,
    const product::DownAndOutOptionParameters*,
    const product::DownAndOutOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);

}  // namespace ai_factory::workbench::model::equity::rough_heston
