// Quadratic rough-Heston phoenix-memory-autocall legacy delta API backed by the rough lift graph.
#include "model/equity/rough/quadratic_rough_heston/product/phoenix_memory_autocall_price_delta.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/phoenix_memory_autocall_price_gradients.cuh"
#include "common/volterra/price_gradients/legacy_spot_delta_graph.cuh"

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

template<std::size_t FactorCount>
void launch_quadratic_rough_heston_phoenix_memory_autocall_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::PhoenixMemoryAutocallParameters* host_products,
    const product::PhoenixMemoryAutocallParameters* device_products,
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
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_standard_errors
) {
    (void)device_models;
    (void)device_products;
    using Graph = PhoenixMemoryAutocallNodeGraph<FactorCount, ::ai_factory::workbench::price_gradients::SensitivityOrders::first, 1U>;
    volterra::price_gradients::launch_legacy_lift_spot_delta<Graph>(
        host_models, model_count, device_prepared_dynamics,
        prepared_dynamics_count, host_products, product_count,
        construction, result_count, result_offset, launch_result_count,
        monte_carlo_paths_per_price, dt, simulation_steps_per_day,
        threads_per_block, block_count, base_seed, bump,
        device_prices, device_standard_errors, device_deltas,
        device_delta_standard_errors,
        prepare_quadratic_rough_heston_phoenix_memory_autocall_sensitivities
    );
}

template void launch_quadratic_rough_heston_phoenix_memory_autocall_price_delta_cuda<
    2U
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const PreparedDynamics<2U>*, std::size_t,
    const product::PhoenixMemoryAutocallParameters*,
    const product::PhoenixMemoryAutocallParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*
);

template void launch_quadratic_rough_heston_phoenix_memory_autocall_price_delta_cuda<
    3U
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const PreparedDynamics<3U>*, std::size_t,
    const product::PhoenixMemoryAutocallParameters*,
    const product::PhoenixMemoryAutocallParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*
);

template void launch_quadratic_rough_heston_phoenix_memory_autocall_price_delta_cuda<
    7U
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const PreparedDynamics<7U>*, std::size_t,
    const product::PhoenixMemoryAutocallParameters*,
    const product::PhoenixMemoryAutocallParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*
);

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
