// Generic path advancement for equity dynamics and observation handlers.
#pragma once

#include "common/equity/concepts.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::equity {

// Terminal state reached through homogeneous fixed-size numerical steps.
template<EquityDynamicsPolicy Dynamics>
__device__ __forceinline__ typename Dynamics::State
simulate_fixed_step_terminal(
    const typename Dynamics::PreparedDynamics& dynamics,
    std::uint32_t transition_count,
    philox::PhiloxKey key,
    std::size_t path
) {
    typename Dynamics::State state = Dynamics::initial_state(dynamics);
    if (transition_count == 0U) return state;
    typename Dynamics::RandomContext random(
        key,
        static_cast<std::uint64_t>(path)
    );
    Dynamics::advance(dynamics, transition_count, random, state);
    return state;
}

// Terminal state reached through one direct transition over the full horizon.
template<ExactTransitionDynamicsPolicy Dynamics>
__device__ __forceinline__ typename Dynamics::State
simulate_exact_transition_terminal(
    const typename Dynamics::PreparedModel& model,
    const typename Dynamics::PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path
) {
    typename Dynamics::State state = Dynamics::initial_state(model);
    typename Dynamics::RandomContext random(
        key,
        static_cast<std::uint64_t>(path)
    );
    Dynamics::simulate_one_step(model, transition, random, state);
    return state;
}

// Regular schedule for a numerical scheme, with an optional shorter initial
// stub. The handler is called only at contractual observation dates.
template<
    EquityDynamicsPolicy Dynamics,
    ObservationHandlerFor<Dynamics> Handler
>
__device__ __forceinline__ typename Dynamics::State
simulate_fixed_step_regular_schedule(
    const typename Dynamics::PreparedDynamics& dynamics,
    std::uint32_t initial_transition_count,
    std::uint32_t transitions_per_observation,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    Handler& handler
) {
    typename Dynamics::State state = Dynamics::initial_state(dynamics);
    if (!handler.on_initial_state(state) || observation_count == 0U) {
        return state;
    }

    typename Dynamics::RandomContext random(
        key,
        static_cast<std::uint64_t>(path)
    );
    Dynamics::advance(
        dynamics,
        initial_transition_count,
        random,
        state
    );
    if (!handler.on_observation(0U, state)) return state;
    for (std::uint32_t observation = 1U;
         observation < observation_count;
         ++observation) {
        Dynamics::advance(
            dynamics,
            transitions_per_observation,
            random,
            state
        );
        if (!handler.on_observation(observation, state)) return state;
    }
    return state;
}

// Regular schedule for a direct-transition model, with a distinct initial
// transition for an optional stub.
template<
    ExactTransitionDynamicsPolicy Dynamics,
    ObservationHandlerFor<Dynamics> Handler
>
__device__ __forceinline__ typename Dynamics::State
simulate_exact_transition_regular_schedule(
    const typename Dynamics::PreparedModel& model,
    const typename Dynamics::PreparedTransition& initial_transition,
    const typename Dynamics::PreparedTransition& regular_transition,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    Handler& handler
) {
    typename Dynamics::State state = Dynamics::initial_state(model);
    if (!handler.on_initial_state(state) || observation_count == 0U) {
        return state;
    }

    typename Dynamics::RandomContext random(
        key,
        static_cast<std::uint64_t>(path)
    );
    Dynamics::simulate_one_step(
        model,
        initial_transition,
        random,
        state
    );
    if (!handler.on_observation(0U, state)) return state;
    for (std::uint32_t observation = 1U;
         observation < observation_count;
         ++observation) {
        Dynamics::simulate_one_step(
            model,
            regular_transition,
            random,
            state
        );
        if (!handler.on_observation(observation, state)) return state;
    }
    return state;
}

// Irregular calendar for a numerical scheme. Each contractual interval stores
// only the number of homogeneous numerical transitions to execute.
template<
    EquityDynamicsPolicy Dynamics,
    ObservationHandlerFor<Dynamics> Handler
>
__device__ __forceinline__ typename Dynamics::State
simulate_fixed_step_calendar(
    const typename Dynamics::PreparedDynamics& dynamics,
    const std::uint32_t* __restrict__ transitions_between_observations,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    Handler& handler
) {
    typename Dynamics::State state = Dynamics::initial_state(dynamics);
    if (!handler.on_initial_state(state) || observation_count == 0U) {
        return state;
    }

    typename Dynamics::RandomContext random(
        key,
        static_cast<std::uint64_t>(path)
    );
    for (std::uint32_t observation = 0U;
         observation < observation_count;
         ++observation) {
        Dynamics::advance(
            dynamics,
            transitions_between_observations[observation],
            random,
            state
        );
        if (!handler.on_observation(observation, state)) return state;
    }
    return state;
}

// Irregular calendar for a direct model. Invariant model coefficients are
// stored once and each contractual interval stores one compact transition.
template<
    ExactTransitionDynamicsPolicy Dynamics,
    ObservationHandlerFor<Dynamics> Handler
>
__device__ __forceinline__ typename Dynamics::State
simulate_exact_transition_calendar(
    const typename Dynamics::PreparedModel& model,
    const typename Dynamics::PreparedTransition* __restrict__ transitions,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    Handler& handler
) {
    typename Dynamics::State state = Dynamics::initial_state(model);
    if (!handler.on_initial_state(state) || observation_count == 0U) {
        return state;
    }

    typename Dynamics::RandomContext random(
        key,
        static_cast<std::uint64_t>(path)
    );
    for (std::uint32_t observation = 0U;
         observation + 1U < observation_count;
         ++observation) {
        Dynamics::simulate_one_step(
            model,
            transitions[observation],
            random,
            state
        );
        if (!handler.on_observation(observation, state)) return state;
    }
    Dynamics::simulate_one_step(
        model,
        transitions[observation_count - 1U],
        random,
        state
    );
    handler.on_observation(observation_count - 1U, state);
    return state;
}

}  // namespace ai_factory::workbench::equity
