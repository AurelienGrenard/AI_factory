// Joint Markovian/rough path scan after causal rough volatility preparation.
#pragma once

#include "common/check_cuda.cuh"
#include "common/multi_asset_price_row.hpp"
#include "common/philox.cuh"
#include "common/reductions.cuh"
#include "common/simulation/multi_asset_path.cuh"
#include "common/volterra/causal_fft_volatility.cuh"
#include "common/volterra/pricing_moments.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

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
__global__ void joint_rough_price_kernel(
    MixedAssetDeviceInputs<MarkovPack, RoughPolicy, Product, RoughCount> inputs,
    std::size_t result_index,
    std::size_t path_offset,
    std::size_t path_count,
    std::uint32_t steps,
    std::uint32_t steps_per_day,
    float dt,
    std::uint64_t base_seed,
    const float* volatility_path,
    volterra::PartialMoments* moments,
    float* path_payoffs
) {
    constexpr std::size_t dimension = MarkovPack::kFactorCount
        + RoughCount * RoughPolicy::kBrownianDimension;
    constexpr std::size_t lower_count = dimension * (dimension + 1U) / 2U;
    constexpr std::size_t rough_spot_offset =
        RoughPolicy::kBrownianDimension == 1U ? 0U
        : RoughCount + MarkovPack::kFactorCount;
    static_assert(TeamSize>=1U && TeamSize<=32U
        && (TeamSize&(TeamSize-1U))==0U);
    constexpr unsigned teams_per_block=128U/TeamSize;
    const unsigned lane=threadIdx.x%TeamSize;
    const unsigned team=threadIdx.x/TeamSize;
    __shared__ typename MarkovPack::Prepared markov_prepared;
    __shared__ typename RoughPolicy::PreparedModel rough_prepared[RoughCount];
    __shared__ typename Product::PreparedProduct prepared_product;
    __shared__ typename Product::ProductParameters product_parameters;
    __shared__ philox::PhiloxKey key;
    __shared__ float lower[lower_count];
    __shared__ typename MarkovPack::States team_states[
        TeamSize>1U ? 128U/TeamSize : 1U];
    if (threadIdx.x == 0U) {
        const auto row = inputs.rows[result_index];
        product_parameters = inputs.products[row.product_index];
        prepared_product = Product::prepare(
            product_parameters, row.discount_factor
        );
        MarkovPack::prepare(
            markov_prepared, inputs.markov_models, row.scenario_index, dt
        );
        for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
            rough_prepared[asset] = RoughPolicy::prepare_model(
                inputs.rough_models[inputs.rough_scenario_indices[
                    static_cast<std::size_t>(row.scenario_index)
                        * RoughCount + asset
                ]], dt
            );
        }
        key = philox::make_key(base_seed + result_index);
    }
    __syncthreads();
    const auto row = inputs.rows[result_index];
    const float* global_lower = inputs.packed_factors
        + static_cast<std::size_t>(row.correlation_index) * lower_count;
    for (std::size_t index = threadIdx.x; index < lower_count;
         index += blockDim.x) lower[index] = global_lower[index];
    __syncthreads();

    double sum=0.0,sumsq=0.0;
    for(unsigned iteration=0U;iteration<TeamSize;++iteration) {
        const auto local_path=blockIdx.x*128U
            +iteration*teams_per_block+team;
        if(local_path>=path_count) continue;
        const std::size_t path = path_offset + local_path;
        typename MarkovPack::States local_states;
        auto& markov_states=TeamSize==1U
            ? local_states : team_states[team];
        MarkovPack::template initial_states<0U, TeamSize>(
            markov_states, markov_prepared, lane
        );
        float rough_log_spots[RoughCount];
        float rough_initial_logs[RoughCount];
        for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
            rough_log_spots[asset] = rough_prepared[asset].initial_log_spot;
            rough_initial_logs[asset] = rough_prepared[asset].initial_log_spot;
        }
        auto handler = Product::make_handler(prepared_product);
        for (std::uint32_t step = 0U; step < steps; ++step) {
            simulation::BrownianNormals<dimension, TeamSize> independent;
            independent.generate(key, path, step, lane);
            simulation::CorrelatedBrownianStep<dimension, TeamSize> brownian;
            brownian.calculate(independent, lower, lane);
            MarkovPack::template advance_one_step<
                0U, RoughCount, dimension, TeamSize
            >(markov_states, markov_prepared, brownian,
              key, path, step, lane);
            const auto slot = volterra::causal_fft::history_index(
                local_path, step, steps
            );
            for (std::size_t asset = 0U; asset < RoughCount; ++asset) {
                const float shock=brownian.get(
                    rough_spot_offset+asset,lane);
                if(asset%TeamSize==lane) {
                    const float variance=volatility_path[
                        asset*path_count*steps+slot];
                    RoughPolicy::advance_spot_cell(
                        rough_prepared[asset],variance,shock,
                        rough_log_spots[asset]);
                }
            }
            if (Product::is_observation_step(
                    step, steps, steps_per_day, product_parameters
                )) {
                const float observation =
                    Product::template observe_joint_group<
                        MarkovPack, RoughPolicy, RoughCount, TeamSize
                    >(
                        markov_states, markov_prepared,
                        rough_log_spots, rough_initial_logs, lane
                    );
                bool keep_going=true;
                if(lane==0U)
                    keep_going=handler.on_observation(step,observation);
                if constexpr (TeamSize>1U)
                    keep_going=__shfl_sync(
                        simulation::subgroup_mask<TeamSize>(),
                        static_cast<int>(keep_going),0U,TeamSize)!=0;
                if(!keep_going) break;
            }
        }
        if(lane==0U) {
            const float payoff=Product::finalize(prepared_product,handler);
            if constexpr (RecordPaths) path_payoffs[path]=payoff;
            sum+=static_cast<double>(payoff);
            sumsq+=static_cast<double>(payoff)*payoff;
        }
    }
    const auto sums=reductions::reduce_block(sum,sumsq);
    if (threadIdx.x == 0U) {
        moments[path_offset / blockDim.x + blockIdx.x] = sums;
    }
}

template<typename MarkovPack, typename RoughPolicy, typename Product,
         bool RecordPaths = false, std::size_t RoughCount = 1U,
         unsigned TeamSize = 1U>
void launch_joint_rough_cuda(
    MixedAssetDeviceInputs<MarkovPack, RoughPolicy, Product, RoughCount> inputs,
    std::size_t result_index,
    std::size_t path_offset,
    std::size_t path_count,
    std::uint32_t steps,
    std::uint32_t steps_per_day,
    float dt,
    std::uint64_t base_seed,
    const float* volatility_path,
    volterra::PartialMoments* moments,
    float* path_payoffs = nullptr
) {
    if (!inputs.rough_models || !inputs.rough_scenario_indices
        || !inputs.products || !inputs.rows || !inputs.packed_factors
        || !volatility_path || !moments || path_count == 0U
        || path_offset % 128U != 0U || steps == 0U || !(dt > 0.0f)) {
        throw std::invalid_argument("invalid joint rough launch");
    }
    if constexpr (RecordPaths) {
        if (!path_payoffs) throw std::invalid_argument(
            "joint rough path output is null");
    }
    joint_rough_price_kernel<
        MarkovPack, RoughPolicy, Product, RecordPaths, RoughCount,
        TeamSize
    >
        <<<static_cast<unsigned int>((path_count + 127U) / 128U),
           128U,8U*sizeof(double)>>>(
            inputs, result_index, path_offset, path_count,
            steps, steps_per_day, dt, base_seed,
            volatility_path, moments, path_payoffs
        );
    check_cuda(cudaGetLastError(), "joint rough price launch");
}

}  // namespace ai_factory::workbench::monte_carlo
