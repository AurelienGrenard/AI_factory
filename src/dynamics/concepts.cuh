// Compile-time contracts for composable stochastic simulation policies.
#pragma once

#include "philox/distributions.cuh"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::simulation {

inline constexpr std::size_t kMaximumObservationHandlerBytes = 128U;

// A path observer only needs to know the state representation. Requiring a
// complete one-step dynamics policy here would incorrectly exclude schemes
// whose state is produced by a block-level Volterra executor.
template<typename StateProvider>
concept StatePolicy =
    std::is_trivially_copyable_v<typename StateProvider::State>;

// Every dynamics exposes model parameters, one path-local random context and
// one mutable state. Market-specific observables are separate capabilities.
template<typename Dynamics>
concept DynamicsPolicy =
    std::is_trivially_copyable_v<typename Dynamics::Parameters>
    && std::is_trivially_copyable_v<typename Dynamics::RandomContext>
    && std::is_trivially_copyable_v<typename Dynamics::State>
    && std::constructible_from<
        typename Dynamics::RandomContext,
        philox::PhiloxKey,
        std::uint64_t
    >;

// Numerical schemes prepare one homogeneous step and expose interval
// advancement as their public customization point. advance(dynamics, n, ...)
// returns the state at the end of an unobserved interval of n homogeneous
// steps. It need not consume the same random stream as n calls with n = 1;
// kPartitionInvariantAdvance advertises that stronger property when it holds.
template<typename Dynamics>
concept PreparedFixedStepDynamicsPolicy =
    DynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<typename Dynamics::PreparedDynamics>
    && requires {
        {
            Dynamics::kPartitionInvariantAdvance
        } -> std::convertible_to<bool>;
    }
    && requires(
        const typename Dynamics::PreparedDynamics& dynamics,
        typename Dynamics::RandomContext& random,
        typename Dynamics::State& state,
        std::uint32_t step_count
    ) {
        {
            Dynamics::initial_state(dynamics)
        } -> std::same_as<typename Dynamics::State>;
        {
            Dynamics::advance(dynamics, step_count, random, state)
        } -> std::same_as<void>;
    };

template<typename Dynamics>
concept FixedStepDynamicsPolicy =
    PreparedFixedStepDynamicsPolicy<Dynamics>
    && requires(
        const typename Dynamics::Parameters& parameters,
        float delta_t
    ) {
        {
            Dynamics::prepare_dynamics(parameters, delta_t)
        } -> std::same_as<typename Dynamics::PreparedDynamics>;
    };

// Direct-transition models keep invariant coefficients separate from the
// coefficients prepared for one interval.
template<typename Dynamics>
concept ExactTransitionDynamicsPolicy =
    DynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<typename Dynamics::PreparedModel>
    && std::is_trivially_copyable_v<typename Dynamics::PreparedTransition>
    && requires(
        const typename Dynamics::Parameters& parameters,
        const typename Dynamics::PreparedModel& model,
        const typename Dynamics::PreparedTransition& transition,
        typename Dynamics::RandomContext& random,
        typename Dynamics::State& state,
        float delta_t
    ) {
        {
            Dynamics::prepare_model(parameters)
        } -> std::same_as<typename Dynamics::PreparedModel>;
        {
            Dynamics::prepare_transition(model, delta_t)
        } -> std::same_as<typename Dynamics::PreparedTransition>;
        {
            Dynamics::initial_state(model)
        } -> std::same_as<typename Dynamics::State>;
        {
            Dynamics::simulate_one_step(
                model,
                transition,
                random,
                state
            )
        } -> std::same_as<void>;
    };

// A handler receives each contractual observation and decides whether the
// path must continue. It is statically dispatched and remains path-local.
template<typename Handler, typename StateProvider>
concept ObservationHandlerFor =
    StatePolicy<StateProvider>
    && std::is_trivially_copyable_v<Handler>
    && sizeof(Handler) <= kMaximumObservationHandlerBytes
    && requires(
        Handler& handler,
        std::uint32_t observation,
        const typename StateProvider::State& state
    ) {
        {
            handler.on_initial_state(state)
        } -> std::same_as<bool>;
        {
            handler.on_observation(observation, state)
        } -> std::same_as<bool>;
    };

template<StatePolicy StateProvider>
struct ObservationHandlerProbe {
    __device__ __forceinline__ bool on_initial_state(
        const typename StateProvider::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename StateProvider::State&
    ) {
        return true;
    }
};

// A scalar observable interprets a model state at the initial point and at
// each contractual observation without imposing any market-specific accessor
// on the dynamics contract.
template<typename Observable, typename StateProvider>
concept ScalarObservableFor =
    StatePolicy<StateProvider>
    && std::is_trivially_copyable_v<Observable>
    && requires(
        const Observable& observable,
        std::uint32_t observation,
        const typename StateProvider::State& state
    ) {
        {
            observable.initial_value(state)
        } -> std::same_as<float>;
        {
            observable.value(observation, state)
        } -> std::same_as<float>;
    };

}  // namespace ai_factory::workbench::simulation
