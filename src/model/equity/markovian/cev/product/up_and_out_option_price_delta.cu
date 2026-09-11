// Generated cev up_and_out_option composition over the shared price-delta engine.
#include "model/equity/markovian/cev/product/up_and_out_option_price_delta.cuh"

#include "common/equity/price_delta/coupled_spot_paths.cuh"
#include "common/equity/price_delta/multiplicative_spot_path.cuh"
#include "common/equity/price_delta/path_product_policy.cuh"
#include "common/monte_carlo/monte_carlo_price_delta_kernel.cuh"
#include "model/equity/markovian/cev/price_delta_dynamics_impl.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::cev {
namespace {
namespace equity = ::ai_factory::workbench::equity;
using Path = equity::price_delta::CoupledSpotPaths<cev::PriceDeltaDynamics>;
using Schedule = simulation::FixedStepDenseSchedule<typename Path::Dynamics>;
template<OptionSide Side>
using PricingPolicy = equity::price_delta::PathProductPriceDeltaPolicy<
    Schedule, product::UpAndOutOptionPathPolicy<Side>, Path>;
static_assert(monte_carlo::PriceDeltaMonteCarloPolicy<PricingPolicy<OptionSide::call>>);
}  // namespace

template<OptionSide Side>
void launch_cev_up_and_out_option_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::UpAndOutOptionParameters* host_products,
    const product::UpAndOutOptionParameters* device_products,
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
    monte_carlo::launch_monte_carlo_price_delta_cuda<PricingPolicy<Side>>(
        {make_model_product_device_inputs(device_models, model_count,
            device_products, product_count, construction), bump},
        {host_models, model_count, host_products, product_count, construction, bump},
        result_count, result_offset, launch_result_count, monte_carlo_paths_per_price,
        {dt, simulation_steps_per_day}, threads_per_block, block_count, base_seed,
        device_prices, device_standard_errors, device_deltas, device_delta_standard_errors,
        "cev.up_and_out_option.price_delta", option_side_name(Side));
}

// Explicit option-side instantiation for the paired-output launcher.
template void launch_cev_up_and_out_option_price_delta_cuda<OptionSide::call>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::UpAndOutOptionParameters*, const product::UpAndOutOptionParameters*,
    std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*);

// Explicit option-side instantiation for the paired-output launcher.
template void launch_cev_up_and_out_option_price_delta_cuda<OptionSide::put>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::UpAndOutOptionParameters*, const product::UpAndOutOptionParameters*,
    std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*);


}  // namespace ai_factory::workbench::model::equity::cev
