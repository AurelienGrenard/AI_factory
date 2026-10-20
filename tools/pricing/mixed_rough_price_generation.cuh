// Homogeneous rough price execution uses the shared heterogeneous runner.
#pragma once

#include "common/monte_carlo/multi_asset_rough_kernel.cuh"
#include "tools/pricing/heterogeneous_rough_price_generation.cuh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

template<typename MarkovPack, typename RoughPolicy, typename Product,
         std::size_t RoughCount = 1U, unsigned TeamSize = 1U>
cuda::MonteCarloRun execute_mixed_rough_prices(
    const MultiAssetHostModels<MarkovPack>& markov_models,
    const std::vector<typename RoughPolicy::Parameters>& rough_models,
    const std::vector<std::uint32_t>& rough_scenario_indices,
    std::size_t scenario_count,
    const std::vector<typename Product::ProductParameters>& products,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows,
    std::size_t paths_per_price, std::size_t path_chunk_size,
    float dt, std::uint32_t steps_per_day,
    std::uint32_t direct_tile_size, std::uint64_t base_seed
) {
    using Pack = simulation::HomogeneousRoughPack<
        RoughPolicy, RoughCount>;
    MultiRoughHostModels<Pack> rough{rough_models, rough_scenario_indices};
    return execute_heterogeneous_rough_prices<
        MarkovPack, Pack, Product, TeamSize>(
            markov_models, rough, scenario_count, products,
            correlations, external_factor_order, internal_factor_order,
            rows, paths_per_price, path_chunk_size, dt, steps_per_day,
            direct_tile_size, base_seed);
}

}  // namespace ai_factory::workbench::offline::pricing
