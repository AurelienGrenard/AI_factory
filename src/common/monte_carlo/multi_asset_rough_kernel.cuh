// Homogeneous rough interface to the shared joint spot scan.
#pragma once

#include "common/monte_carlo/multi_asset_heterogeneous_rough_kernel.cuh"

namespace ai_factory::workbench::monte_carlo {

template<typename MarkovPack, typename RoughPolicy, typename Product,
         std::size_t RoughCount = 1U>
struct MixedAssetDeviceInputs {
    typename MarkovPack::Inputs markov_models;
    const typename RoughPolicy::Parameters* rough_models;
    const std::uint32_t* rough_scenario_indices;
    const typename Product::ProductParameters* products;
    const MultiAssetPriceRow* rows;
    const float* packed_factors;
};

template<typename MarkovPack, typename RoughPolicy, typename Product,
         bool RecordPaths = false, std::size_t RoughCount = 1U,
         unsigned TeamSize = 1U>
void launch_joint_rough_cuda(
    MixedAssetDeviceInputs<MarkovPack, RoughPolicy, Product, RoughCount> inputs,
    std::size_t result_index, std::size_t path_offset,
    std::size_t path_count, std::uint32_t steps,
    std::uint32_t steps_per_day, float dt, std::uint64_t base_seed,
    const float* volatility_path, volterra::PartialMoments* moments,
    float* path_payoffs = nullptr
) {
    using RoughPack = simulation::HomogeneousRoughPack<
        RoughPolicy, RoughCount>;
    HeterogeneousRoughDeviceInputs<MarkovPack, RoughPack, Product>
        shared_inputs{
            inputs.markov_models,
            {inputs.rough_models, inputs.rough_scenario_indices},
            inputs.products, inputs.rows, inputs.packed_factors};
    launch_heterogeneous_rough_cuda<
        MarkovPack, RoughPack, Product, RecordPaths, TeamSize>(
            shared_inputs, result_index, path_offset, path_count,
            steps, steps_per_day, dt, base_seed, volatility_path,
            moments, path_payoffs);
}

}  // namespace ai_factory::workbench::monte_carlo
