// Equity slots whose carry is supplied by one simulated short-rate path.
#pragma once

#include "common/simulation/multi_asset_path.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::simulation {

template<typename DynamicsPolicy, std::size_t Count>
struct RateAdjustedModelGroup : ModelGroup<DynamicsPolicy, Count> {
    static_assert(RateAdjustedCorrelatedStepDynamicsPolicy<
        DynamicsPolicy, philox::DomainUniformSequence>);
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
                Dynamics::nominal_rate_step(input.models[index], dt);
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
        const auto apply = [&](
            std::size_t asset,
            const typename Dynamics::PreparedDynamics& model,
            const typename Dynamics::BrownianStep& shock,
            philox::DomainUniformSequence& auxiliary,
            typename Dynamics::State& state
        ) {
            const float carry_shift =
                rate_integral_step - prepared.nominal_rate_dt[asset];
            Dynamics::one_step_transition_with_rate(
                model, shock, auxiliary, state, carry_shift);
        };
        Base::template advance_one_step_with_transition<
            AssetOffset, FactorOffset, Dimension, TeamSize>(
                states, prepared.equity, brownian, key, path,
                step, lane, apply);
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
