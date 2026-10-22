// Contracts for prepared simulation schedules.
#pragma once

#include "dynamics/concepts.cuh"
#include "schedule/calendar.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::simulation {

// Every schedule prepares one simulation grid and validates its numerical
// time configuration. Path observation and terminal simulation are narrower
// capabilities layered below.
template<typename Schedule>
concept DevicePreparedSchedulePolicy = requires(
    const typename Schedule::Dynamics::Parameters& parameters,
    const typename Schedule::Calendar& calendar,
    const typename Schedule::TimeConfiguration& time_configuration
) {
    {
        Schedule::prepare(parameters, calendar, time_configuration)
    } -> std::same_as<typename Schedule::PreparedSchedule>;
};

template<typename Schedule>
concept ExternallyPreparedSchedulePolicy = requires(
    const typename Schedule::Dynamics::PreparedDynamics& dynamics,
    const typename Schedule::Calendar& calendar,
    const typename Schedule::TimeConfiguration& time_configuration
) {
    {
        Schedule::prepare_from_dynamics(
            dynamics, calendar, time_configuration
        )
    } -> std::same_as<typename Schedule::PreparedSchedule>;
};

// Model-only sampling may reuse parameter-dependent coefficients for every
// path in a parameter block while still preparing the calendar-dependent
// part per sample.  The neutral PreparedInput name covers both a fixed-step
// PreparedDynamics and an exact-transition PreparedModel.
template<typename Schedule>
concept ReusablePreparedInputSchedulePolicy =
    std::is_trivially_copyable_v<typename Schedule::PreparedInput>
    && requires(
        const typename Schedule::Dynamics::Parameters& parameters,
        const typename Schedule::PreparedInput& prepared_input,
        const typename Schedule::Calendar& calendar,
        const typename Schedule::TimeConfiguration& time_configuration
    ) {
        {
            Schedule::prepare_input(parameters, time_configuration)
        } -> std::same_as<typename Schedule::PreparedInput>;
        {
            Schedule::prepare_from_input(
                prepared_input, calendar, time_configuration
            )
        } -> std::same_as<typename Schedule::PreparedSchedule>;
    };

template<typename Schedule>
concept SchedulePolicy =
    DynamicsPolicy<typename Schedule::Dynamics>
    && std::is_trivially_copyable_v<typename Schedule::TimeConfiguration>
    && std::is_trivially_copyable_v<typename Schedule::Calendar>
    && std::is_trivially_copyable_v<typename Schedule::PreparedSchedule>
    && requires(
        const typename Schedule::TimeConfiguration& time_configuration
    ) {
        {
            ::ai_factory::workbench::simulation::
                validate_time_configuration(
                    time_configuration
                )
        } -> std::same_as<void>;
    }
    && (DevicePreparedSchedulePolicy<Schedule>
        || ExternallyPreparedSchedulePolicy<Schedule>);

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
