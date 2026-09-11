// Adapt a model-owned three-state transition to the shared price-delta observer.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"

namespace ai_factory::workbench::equity::price_delta {

template<typename CoupledDynamics>
struct CoupledSpotPaths {
    using Dynamics = CoupledDynamics;
    using ModelParameters = typename Dynamics::ModelParameters;
    struct Prepared {};

    __device__ __forceinline__ static Prepared prepare(SpotBump) { return {}; }

    __device__ __forceinline__ static typename Dynamics::Parameters parameters(
        const ModelParameters& model, SpotBump bump
    ) {
        return {model, bump.lower, bump.upper};
    }

    template<unsigned int Scenario>
    __device__ __forceinline__ static SpotObservation observe(
        const Prepared&, const typename Dynamics::State& state
    ) {
        return Dynamics::template observe<Scenario>(state);
    }
};

}  // namespace ai_factory::workbench::equity::price_delta
