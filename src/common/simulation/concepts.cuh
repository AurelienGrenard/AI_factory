// Compile-time contracts for composable stochastic simulation policies.
#pragma once

#include "common/philox.cuh"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::simulation {

inline constexpr std::size_t kMaximumObservationHandlerBytes = 128U;

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
// advancement as their public customization point.
template<typename Dynamics>
concept FixedStepDynamicsPolicy =
    DynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<typename Dynamics::PreparedDynamics>
    && requires(
        const typename Dynamics::Parameters& parameters,
        const typename Dynamics::PreparedDynamics& dynamics,
        typename Dynamics::RandomContext& random,
        typename Dynamics::State& state,
        std::uint32_t step_count,
        float delta_t
    ) {
        {
            Dynamics::prepare_dynamics(parameters, delta_t)
        } -> std::same_as<typename Dynamics::PreparedDynamics>;
        {
            Dynamics::initial_state(dynamics)
        } -> std::same_as<typename Dynamics::State>;
        {
            Dynamics::advance(dynamics, step_count, random, state)
        } -> std::same_as<void>;
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
template<typename Handler, typename Dynamics>
concept ObservationHandlerFor =
    DynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<Handler>
    && sizeof(Handler) <= kMaximumObservationHandlerBytes
    && requires(
        Handler& handler,
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        {
            handler.on_initial_state(state)
        } -> std::same_as<bool>;
        {
            handler.on_observation(observation, state)
        } -> std::same_as<bool>;
    };

template<DynamicsPolicy Dynamics>
struct ObservationHandlerProbe {
    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State&
    ) {
        return true;
    }
};

// A scalar observable interprets a model state at the initial point and at
// each contractual observation without imposing any market-specific accessor
// on the dynamics contract.
template<typename Observable, typename Dynamics>
concept ScalarObservableFor =
    DynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<Observable>
    && requires(
        const Observable& observable,
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        {
            observable.initial_value(state)
        } -> std::same_as<float>;
        {
            observable.value(observation, state)
        } -> std::same_as<float>;
    };

// Every schedule prepares one simulation grid and validates its numerical
// time configuration. Path observation and terminal simulation are narrower
// capabilities layered below.
template<typename Schedule>
concept SchedulePolicy =
    DynamicsPolicy<typename Schedule::Dynamics>
    && std::is_trivially_copyable_v<typename Schedule::TimeConfiguration>
    && std::is_trivially_copyable_v<typename Schedule::Calendar>
    && std::is_trivially_copyable_v<typename Schedule::PreparedSchedule>
    && requires(
        const typename Schedule::Dynamics::Parameters& parameters,
        const typename Schedule::Calendar& calendar,
        const typename Schedule::TimeConfiguration& time_configuration
    ) {
        {
            Schedule::prepare(
                parameters,
                calendar,
                time_configuration
            )
        } -> std::same_as<typename Schedule::PreparedSchedule>;
        {
            validate_time_configuration(time_configuration)
        } -> std::same_as<void>;
    };

// Path-dependent products receive contractual observations through a handler.
template<typename Schedule>
concept ObservedSchedulePolicy =
    SchedulePolicy<Schedule>
    && requires(
        const typename Schedule::PreparedSchedule& prepared,
        philox::PhiloxKey key,
        std::size_t path,
        ObservationHandlerProbe<typename Schedule::Dynamics>& handler
    ) {
        {
            Schedule::simulate(prepared, key, path, handler)
        } -> std::same_as<typename Schedule::Dynamics::State>;
    };

// Some observed products need the number of contractual observations prepared
// by the schedule.
template<typename Schedule>
concept CountedObservedSchedulePolicy =
    ObservedSchedulePolicy<Schedule>
    && requires(const typename Schedule::PreparedSchedule& prepared) {
        {
            Schedule::observation_count(prepared)
        } -> std::same_as<std::uint32_t>;
    };

// Dense schedules expose every homogeneous numerical transition to the
// observation handler.
template<typename Schedule>
concept DenseSchedulePolicy =
    CountedObservedSchedulePolicy<Schedule>
    && requires {
        { Schedule::kObservesEveryTransition } -> std::convertible_to<bool>;
    }
    && Schedule::kObservesEveryTransition;

// Terminal-only products bypass the observation lifecycle and request the
// final state directly from either a fixed-step or exact-transition schedule.
template<typename Schedule>
concept TerminalSchedulePolicy =
    SchedulePolicy<Schedule>
    && requires(
        const typename Schedule::PreparedSchedule& prepared,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        {
            Schedule::simulate_terminal(prepared, key, path)
        } -> std::same_as<typename Schedule::Dynamics::State>;
    };

// Two-date products require exactly two contractual observations without
// depending on the concrete fixed-step or exact-transition schedule type.
template<typename Schedule>
concept TwoDateSchedulePolicy =
    ObservedSchedulePolicy<Schedule>
    && requires {
        { Schedule::kObservationCount } -> std::convertible_to<std::size_t>;
    }
    && Schedule::kObservationCount == 2U;

}  // namespace ai_factory::workbench::simulation
