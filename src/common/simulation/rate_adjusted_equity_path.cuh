// Equity slots whose carry is supplied by one simulated short-rate path.
#pragma once

#include "common/simulation/multi_asset_path.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::simulation {

template<typename DynamicsPolicy, std::size_t Count>
struct RateAdjustedModelGroup : ModelGroup<DynamicsPolicy, Count> {
    using Base = ModelGroup<DynamicsPolicy, Count>;
    using Dynamics = typename Base::Dynamics;
    using Parameters = typename Base::Parameters;
    using Inputs = typename Base::Inputs;
    using States = typename Base::States;
    static constexpr std::size_t kAssetCount = Base::kAssetCount;
    static constexpr std::size_t kFactorCount = Base::kFactorCount;

    struct Prepared {
        typename Base::Prepared equity;
        float nominal_rate_dt[Count];
    };

    __device__ __forceinline__ static void prepare(
        Prepared& output, const Inputs& input,
        std::uint32_t scenario, float dt
    ) {
        Base::prepare(output.equity, input, scenario, dt);
        for (std::size_t asset = 0U; asset < Count; ++asset) {
            const auto index = input.scenario_indices[scenario * Count + asset];
            output.nominal_rate_dt[asset] =
                input.models[index].risk_free_rate * dt;
        }
    }

    template<std::size_t AssetOffset, unsigned int TeamSize>
    __device__ __forceinline__ static void initial_states(
        States& states, const Prepared& prepared, unsigned int lane
    ) {
        Base::template initial_states<AssetOffset, TeamSize>(
            states, prepared.equity, lane);
    }

    template<std::size_t AssetOffset, std::size_t FactorOffset,
             std::size_t Dimension, unsigned int TeamSize>
    __device__ __forceinline__ static void advance_one_step(
        States& states, const Prepared& prepared,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian,
        philox::PhiloxKey key, std::uint64_t path,
        std::uint32_t step, unsigned int lane
    ) {
        Base::template advance_one_step<
            AssetOffset, FactorOffset, Dimension, TeamSize>(
                states, prepared.equity, brownian, key, path, step, lane);
    }

    template<std::size_t AssetOffset, std::size_t FactorOffset,
             std::size_t Dimension, unsigned int TeamSize>
    __device__ __forceinline__ static void advance_one_step_with_rate(
        States& states, const Prepared& prepared,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian,
        philox::PhiloxKey key, std::uint64_t path,
        std::uint32_t step, unsigned int lane, float rate_integral_step
    ) {
        static_assert(AssetOffset + Count + 1U <= 255U);
        const auto innovation=[&](std::size_t asset) {
            const std::size_t first=FactorOffset
                +asset*Dynamics::kBrownianDimension;
            const float spot=brownian.get(first,lane);
            if constexpr (Dynamics::kBrownianDimension==1U)
                return typename Dynamics::BrownianStep{spot};
            else return typename Dynamics::BrownianStep{
                spot,brownian.get(first+1U,lane)};
        };
        const auto transition=[&](
            std::size_t asset,const typename Dynamics::BrownianStep& shock
        ) {
            const auto source=static_cast<std::uint32_t>(
                2U+AssetOffset+asset);
            philox::DomainUniformSequence auxiliary(
                key,path,(source<<24U)|step);
            const float shift=rate_integral_step
                -prepared.nominal_rate_dt[asset];
            if constexpr (requires {
                    prepared.equity.model[asset].carry_time_step;
                }) {
                auto adjusted=prepared.equity.model[asset];
                adjusted.carry_time_step+=shift;
                Dynamics::one_step_transition(
                    adjusted,shock,auxiliary,states.model[asset]);
            } else if constexpr (requires {
                    states.model[asset].spot;
                }) {
                auto adjusted=prepared.equity.model[asset];
                adjusted.drift_dt+=shift;
                Dynamics::one_step_transition(
                    adjusted,shock,auxiliary,states.model[asset]);
            } else {
                Dynamics::one_step_transition(
                    prepared.equity.model[asset],shock,auxiliary,
                    states.model[asset]);
                states.model[asset].log_spot+=shift;
            }
        };
        bool distributed=TeamSize==1U;
        if constexpr (TeamSize>1U)
            distributed=brownian.has_shared_factors();
        if(distributed) {
            constexpr std::size_t lane_shift=AssetOffset%TeamSize;
            constexpr std::size_t local_count=
                (Count+TeamSize-1U)/TeamSize;
            for(std::size_t local=0U;local<local_count;++local) {
                const std::size_t asset=local*TeamSize
                    +(lane+TeamSize-lane_shift)%TeamSize;
                const auto shock=innovation(asset<Count ? asset : 0U);
                if(asset<Count) transition(asset,shock);
            }
        } else {
            for(std::size_t asset=0U;asset<Count;++asset) {
                const auto shock=innovation(asset);
                if((AssetOffset+asset)%TeamSize==lane)
                    transition(asset,shock);
                if constexpr (TeamSize>1U)
                    __syncwarp(subgroup_mask<TeamSize>());
            }
        }
    }

    template<std::size_t AssetOffset, unsigned int TeamSize>
    __device__ __forceinline__ static float minimum_ratio(
        const States& states, const Prepared& prepared, unsigned int lane
    ) {
        return Base::template minimum_ratio<AssetOffset, TeamSize>(
            states, prepared.equity, lane);
    }

    template<std::size_t AssetOffset, unsigned int TeamSize>
    __device__ __forceinline__ static float sum_ratio(
        const States& states, const Prepared& prepared, unsigned int lane
    ) {
        return Base::template sum_ratio<AssetOffset, TeamSize>(
            states, prepared.equity, lane);
    }
};

}  // namespace ai_factory::workbench::simulation
