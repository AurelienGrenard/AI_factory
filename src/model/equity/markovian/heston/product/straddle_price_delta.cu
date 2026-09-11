// Generated heston straddle composition over the shared price-delta engine.
#include "model/equity/markovian/heston/product/straddle_price_delta.cuh"

#include "common/equity/price_delta/coupled_spot_paths.cuh"
#include "common/equity/price_delta/multiplicative_spot_path.cuh"
#include "common/equity/price_delta/path_product_policy.cuh"
#include "common/monte_carlo/monte_carlo_price_delta_kernel.cuh"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "product/straddle/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::heston {
namespace {
namespace equity = ::ai_factory::workbench::equity;
using Path = equity::price_delta::MultiplicativeSpotPath<heston::DynamicsPolicy>;
using Schedule = simulation::FixedStepTerminalSchedule<typename Path::Dynamics>;

using PricingPolicy = equity::price_delta::PathProductPriceDeltaPolicy<
    Schedule, product::StraddlePathPolicy, Path>;
static_assert(monte_carlo::PriceDeltaMonteCarloPolicy<PricingPolicy>);
}  // namespace


void launch_heston_straddle_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::StraddleParameters* host_products,
    const product::StraddleParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt, std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_standard_errors
) {
    monte_carlo::launch_monte_carlo_price_delta_cuda<PricingPolicy>(
        {make_model_product_device_inputs(device_models, model_count,
            device_products, product_count, construction), bump},
        {host_models, model_count, host_products, product_count, construction, bump},
        result_count, result_offset, launch_result_count, monte_carlo_paths_per_price,
        {dt, simulation_steps_per_day}, threads_per_block, block_count, base_seed,
        device_prices, device_standard_errors, device_deltas, device_delta_standard_errors,
        "heston.straddle.price_delta", "default");
}



}  // namespace ai_factory::workbench::model::equity::heston
