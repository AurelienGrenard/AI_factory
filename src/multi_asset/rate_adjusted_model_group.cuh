// Equity slots whose carry is supplied by one simulated short-rate path.
#pragma once

#include "multi_asset/markovian_pack.cuh"

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
    using States = typename Base::States;
    using Prepared = typename Base::Prepared;

    template<std::size_t AssetOffset, std::size_t FactorOffset,
             std::size_t Dimension, unsigned int TeamSize>
    __device__ __forceinline__ static void advance_one_step_with_rate(
        States& states, const Prepared& prepared,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian,
        philox::PhiloxKey key, std::uint64_t path,
        std::uint32_t step, unsigned int lane, float rate_integral_step
    ) {
        const auto apply = [&](
            std::size_t,
            const typename Dynamics::PreparedDynamics& model,
            const typename Dynamics::BrownianStep& shock,
            philox::DomainUniformSequence& auxiliary,
            typename Dynamics::State& state
        ) {
            Dynamics::one_step_transition_with_rate(
                model, shock, auxiliary, state, rate_integral_step);
        };
        Base::template advance_one_step_with_transition<
            AssetOffset, FactorOffset, Dimension, TeamSize>(
                states, prepared, brownian, key, path,
                step, lane, apply);
    }
};

}  // namespace ai_factory::workbench::simulation
