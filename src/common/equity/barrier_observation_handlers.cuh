// Path-local barrier state shared by discretely monitored equity products.
#pragma once

#include "common/equity/concepts.cuh"

#include <cstdint>

namespace ai_factory::workbench::equity {

enum class BarrierDirection : std::uint8_t {
    down,
    up,
};

template<BarrierDirection Direction>
__device__ __forceinline__ bool barrier_breached(
    float spot,
    float barrier
) {
    if constexpr (Direction == BarrierDirection::down) {
        return spot <= barrier;
    } else {
        return spot >= barrier;
    }
}

template<EquityDynamicsPolicy Dynamics, BarrierDirection Direction>
struct KnockOutBarrierObservationHandler {
    float barrier;
    float terminal_spot = 0.0f;

    __device__ __forceinline__ bool observe(
        const typename Dynamics::State& state
    ) {
        terminal_spot = Dynamics::spot(state);
        return !barrier_breached<Direction>(terminal_spot, barrier);
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return observe(state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        return observe(state);
    }
};

template<EquityDynamicsPolicy Dynamics, BarrierDirection Direction>
struct KnockInBarrierObservationHandler {
    float barrier;
    bool activated = false;

    __device__ __forceinline__ bool observe(
        const typename Dynamics::State& state
    ) {
        if (!activated) {
            activated = barrier_breached<Direction>(
                Dynamics::spot(state),
                barrier
            );
        }
        return true;
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return observe(state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        return observe(state);
    }
};

template<EquityDynamicsPolicy Dynamics>
struct DoubleKnockOutObservationHandler {
    float lower_barrier;
    float upper_barrier;
    float terminal_spot = 0.0f;

    __device__ __forceinline__ bool observe(
        const typename Dynamics::State& state
    ) {
        terminal_spot = Dynamics::spot(state);
        return terminal_spot > lower_barrier
            && terminal_spot < upper_barrier;
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return observe(state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        return observe(state);
    }
};

}  // namespace ai_factory::workbench::equity
