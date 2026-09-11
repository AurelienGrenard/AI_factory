// Reuse a central multiplicative path, including its variance and jump draws.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"

namespace ai_factory::workbench::equity::price_delta {

// Instantiate only for a model/scheme whose S0 homogeneity is qualified.
template<typename ModelDynamics>
struct MultiplicativeSpotPath {
    using ModelParameters = typename ModelDynamics::Parameters;
    using Dynamics = ModelDynamics;
    struct Prepared {
        float lower_scale;
        float upper_scale;
        float lower_log_shift;
        float upper_log_shift;
    };

    __device__ __forceinline__ static Prepared prepare(SpotBump bump) {
        const float lower_scale = bump.lower / bump.central;
        const float upper_scale = bump.upper / bump.central;
        return {lower_scale, upper_scale, logf(lower_scale), logf(upper_scale)};
    }

    __device__ __forceinline__ static ModelParameters parameters(
        const ModelParameters& model, SpotBump
    ) {
        return model;
    }

    template<unsigned int Scenario>
    __device__ __forceinline__ static SpotObservation observe(
        const Prepared& prepared, const typename Dynamics::State& state
    ) {
        static_assert(Scenario < 3U);
        const float spot = Dynamics::spot(state);
        const float log_spot = Dynamics::log_spot(state);
        if constexpr (Scenario == 0U) {
            return {spot, log_spot};
        } else if constexpr (Scenario == 1U) {
            return {spot * prepared.lower_scale, log_spot + prepared.lower_log_shift};
        } else {
            return {spot * prepared.upper_scale, log_spot + prepared.upper_log_shift};
        }
    }
};

}  // namespace ai_factory::workbench::equity::price_delta
