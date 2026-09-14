// Public Quadratic rough-Heston Straddle N-factor price/spot-delta launcher.

#include "model/equity/rough/quadratic_rough_heston/product/straddle_price_delta.cuh"
#include "common/equity/price_delta/path_product_policy.cuh"
#include "common/equity/price_delta/multiplicative_spot_path.cuh"
#include "common/monte_carlo/monte_carlo_price_delta_kernel.cuh"
#include "model/equity/rough/quadratic_rough_heston/dynamics_impl.cuh"
#include "product/straddle/pricing_policy.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {
namespace {
template<std::size_t FactorCount>
using Schedule = simulation::FixedStepTerminalSchedule<DynamicsPolicy<FactorCount>>;
}  // namespace

template<std::size_t FactorCount>
void launch_quadratic_rough_heston_straddle_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::StraddleParameters* host_products,
    const product::StraddleParameters* device_products,
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
    namespace delta = ::ai_factory::workbench::equity::price_delta;
    using Product = product::StraddlePathPolicy;
    using Inputs = PreparedModelProductDeviceInputs<ModelParameters,
        typename Product::ProductParameters, PreparedDynamics<FactorCount>>;
    using Policy = delta::PathProductPriceDeltaPolicy<Schedule<FactorCount>,
        Product, delta::MultiplicativeSpotPath<DynamicsPolicy<FactorCount>>, Inputs>;
    const auto primary = make_prepared_model_product_device_inputs(
        device_models, model_count, device_products, product_count, construction,
        device_prepared_dynamics, prepared_dynamics_count);
    monte_carlo::launch_monte_carlo_price_delta_cuda<Policy>(
        {primary, bump},
        {host_models, model_count, host_products, product_count, construction, bump},
        result_count, result_offset, launch_result_count, monte_carlo_paths_per_price,
        simulation::FixedStepTimeConfiguration{dt, simulation_steps_per_day},
        threads_per_block, block_count, base_seed, device_prices, device_standard_errors,
        device_deltas, device_delta_standard_errors,
        "quadratic_rough_heston.straddle_price_delta", "default");
}

template void launch_quadratic_rough_heston_straddle_price_delta_cuda<
    2U
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const PreparedDynamics<2U>*, std::size_t,
    const product::StraddleParameters*,
    const product::StraddleParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*
);

template void launch_quadratic_rough_heston_straddle_price_delta_cuda<
    3U
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const PreparedDynamics<3U>*, std::size_t,
    const product::StraddleParameters*,
    const product::StraddleParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*
);

template void launch_quadratic_rough_heston_straddle_price_delta_cuda<
    7U
>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const PreparedDynamics<7U>*, std::size_t,
    const product::StraddleParameters*,
    const product::StraddleParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*
);

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
