// Path-local barrier state shared by discretely monitored scalar observables.
#pragma once

#include "common/payoff/barrier.cuh"
#include "common/simulation/concepts.cuh"

#include <cstdint>

namespace ai_factory::workbench::simulation {

template<
    DynamicsPolicy Dynamics,
    typename Observable,
    payoff::BarrierDirection Direction
>
requires ScalarObservableFor<Observable, Dynamics>
struct KnockOutBarrierObservationHandler {
    float barrier;
    float terminal_value = 0.0f;
    [[no_unique_address]] Observable observable{};

    __device__ __forceinline__ bool observe(
        float value
    ) {
        terminal_value = value;
        return !payoff::barrier_breached<Direction>(
            terminal_value,
            barrier
        );
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return observe(observable.initial_value(state));
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        return observe(observable.value(observation, state));
    }
};

template<
    DynamicsPolicy Dynamics,
    typename Observable,
    payoff::BarrierDirection Direction
>
requires ScalarObservableFor<Observable, Dynamics>
struct KnockInBarrierObservationHandler {
    float barrier;
    bool activated = false;
    [[no_unique_address]] Observable observable{};

    __device__ __forceinline__ bool observe(
        float value
    ) {
        if (!activated) {
            activated = payoff::barrier_breached<Direction>(
                value,
                barrier
            );
        }
        return true;
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return observe(observable.initial_value(state));
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        return observe(observable.value(observation, state));
    }
};

template<DynamicsPolicy Dynamics, typename Observable>
requires ScalarObservableFor<Observable, Dynamics>
struct DoubleKnockOutObservationHandler {
    float lower_barrier;
    float upper_barrier;
    float terminal_value = 0.0f;
    [[no_unique_address]] Observable observable{};

    __device__ __forceinline__ bool observe(
        float value
    ) {
        terminal_value = value;
        return terminal_value > lower_barrier
            && terminal_value < upper_barrier;
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return observe(observable.initial_value(state));
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        return observe(observable.value(observation, state));
    }
};

}  // namespace ai_factory::workbench::simulation
