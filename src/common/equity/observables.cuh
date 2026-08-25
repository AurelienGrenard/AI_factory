// Scalar observables extracted from equity model states.
#pragma once

#include "common/equity/concepts.cuh"

#include <cstdint>

namespace ai_factory::workbench::equity {

template<SpotDynamicsPolicy Dynamics>
struct SpotObservable {
    __device__ __forceinline__ float initial_value(
        const typename Dynamics::State& state
    ) const {
        return Dynamics::spot(state);
    }

    __device__ __forceinline__ float value(
        std::uint32_t,
        const typename Dynamics::State& state
    ) const {
        return Dynamics::spot(state);
    }
};

}  // namespace ai_factory::workbench::equity
