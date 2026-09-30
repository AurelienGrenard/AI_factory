// Compact schedule metadata shared by mono and node-graph path sensitivities.
#pragma once

#include "common/price_gradients/time_configuration.hpp"
#include "common/simulation/schedule.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

enum class PathScheduleKind : std::uint8_t {
    dense,
    regular,
    calendar,
};

template<typename Schedule>
struct PathScheduleTraits;

template<typename CanonicalDynamics>
struct PathScheduleTraits<
    simulation::FixedStepDenseSchedule<CanonicalDynamics>
> {
    using Schedule = simulation::FixedStepDenseSchedule<CanonicalDynamics>;
    using Calendar = typename Schedule::Calendar;
    static constexpr PathScheduleKind kKind = PathScheduleKind::dense;
    static constexpr bool kExactTransition = false;
    static constexpr std::size_t kIntervalCapacity = 1U;
};

template<typename CanonicalDynamics>
struct PathScheduleTraits<
    simulation::FixedStepRegularSchedule<CanonicalDynamics>
> {
    using Schedule = simulation::FixedStepRegularSchedule<CanonicalDynamics>;
    using Calendar = typename Schedule::Calendar;
    static constexpr PathScheduleKind kKind = PathScheduleKind::regular;
    static constexpr bool kExactTransition = false;
    static constexpr std::size_t kIntervalCapacity = 1U;
};

template<typename CanonicalDynamics>
struct PathScheduleTraits<
    simulation::ExactTransitionRegularSchedule<CanonicalDynamics>
> {
    using Schedule = simulation::ExactTransitionRegularSchedule<CanonicalDynamics>;
    using Calendar = typename Schedule::Calendar;
    static constexpr PathScheduleKind kKind = PathScheduleKind::regular;
    static constexpr bool kExactTransition = true;
    static constexpr std::size_t kIntervalCapacity = 1U;
};

template<typename CanonicalDynamics, std::size_t ObservationCount>
struct PathScheduleTraits<
    simulation::FixedStepCalendarSchedule<
        CanonicalDynamics, ObservationCount
    >
> {
    using Schedule = simulation::FixedStepCalendarSchedule<
        CanonicalDynamics, ObservationCount
    >;
    using Calendar = typename Schedule::Calendar;
    static constexpr PathScheduleKind kKind = PathScheduleKind::calendar;
    static constexpr bool kExactTransition = false;
    static constexpr std::size_t kIntervalCapacity = ObservationCount;
};

template<typename CanonicalDynamics, std::size_t ObservationCount>
struct PathScheduleTraits<
    simulation::ExactTransitionCalendarSchedule<
        CanonicalDynamics, ObservationCount
    >
> {
    using Schedule = simulation::ExactTransitionCalendarSchedule<
        CanonicalDynamics, ObservationCount
    >;
    using Calendar = typename Schedule::Calendar;
    static constexpr PathScheduleKind kKind = PathScheduleKind::calendar;
    static constexpr bool kExactTransition = true;
    static constexpr std::size_t kIntervalCapacity = ObservationCount;
};

template<std::size_t IntervalCapacity>
struct PreparedPathSchedule {
    std::uint32_t transition_counts[IntervalCapacity]{};
    float interval_years[IntervalCapacity]{};
    std::uint32_t observation_count = 0U;
};

__host__ __device__ inline float path_day_fraction(
    pg::TimeConfiguration time
) {
    return static_cast<float>(time.simulation_steps_per_day) * time.dt;
}

template<typename Schedule>
__device__ __forceinline__ auto prepare_path_schedule(
    const typename PathScheduleTraits<Schedule>::Calendar& calendar,
    pg::TimeConfiguration time
) {
    using Traits = PathScheduleTraits<Schedule>;
    PreparedPathSchedule<Traits::kIntervalCapacity> result{};
    const float day_fraction = path_day_fraction(time);
    if constexpr (Traits::kKind == PathScheduleKind::dense) {
        result.transition_counts[0U] =
            time.simulation_steps_per_day * calendar.maturity_days;
        result.interval_years[0U] = time.dt;
        result.observation_count = result.transition_counts[0U];
    } else if constexpr (Traits::kKind == PathScheduleKind::regular) {
        result.transition_counts[0U] = Traits::kExactTransition
            ? 1U
            : time.simulation_steps_per_day
                * calendar.observation_interval_days;
        result.interval_years[0U] =
            static_cast<float>(calendar.observation_interval_days)
            * day_fraction;
        result.observation_count = calendar.observation_count;
    } else {
        #pragma unroll
        for (std::size_t interval = 0U;
             interval < Traits::kIntervalCapacity;
             ++interval) {
            result.transition_counts[interval] = Traits::kExactTransition
                ? 1U
                : time.simulation_steps_per_day
                    * calendar.interval_days[interval];
            result.interval_years[interval] =
                static_cast<float>(calendar.interval_days[interval])
                * day_fraction;
        }
        result.observation_count =
            static_cast<std::uint32_t>(Traits::kIntervalCapacity);
    }
    return result;
}


template<std::size_t IntervalCapacity>
struct PreparedTerminalPathSchedule {
    PreparedPathSchedule<IntervalCapacity> central;
    std::uint32_t prefix_observation_count = 0U;
    std::uint32_t prefix_step_count = 0U;
    float prefix_years = 0.0f;
};

struct PreparedTerminalPathNode {
    std::uint32_t transition_count;
    float interval_years;
};

template<typename Schedule>
__device__ __forceinline__ auto prepare_terminal_path_schedule(
    const typename PathScheduleTraits<Schedule>::Calendar& calendar,
    pg::TimeConfiguration time,
    std::uint32_t central_step_count
) {
    using Traits = PathScheduleTraits<Schedule>;
    PreparedTerminalPathSchedule<Traits::kIntervalCapacity> result{};
    result.central = prepare_path_schedule<Schedule>(calendar, time);
    result.prefix_observation_count =
        result.central.observation_count - 1U;
    if constexpr (Traits::kKind == PathScheduleKind::dense) {
        result.prefix_step_count = central_step_count - 1U;
        result.prefix_years =
            static_cast<float>(result.prefix_step_count) * time.dt;
    } else {
        const auto prefix_days =
            simulation::calendar_terminal_prefix_days(calendar);
        result.prefix_step_count = static_cast<std::uint32_t>(
            prefix_days * static_cast<std::uint64_t>(
                time.simulation_steps_per_day
            )
        );
        result.prefix_years = static_cast<float>(result.prefix_step_count)
            * time.dt;
    }
    return result;
}

template<typename Schedule, typename Scenario>
__device__ __forceinline__ PreparedTerminalPathNode
prepare_terminal_path_node(
    const Scenario& scenario,
    const PreparedTerminalPathSchedule<
        PathScheduleTraits<Schedule>::kIntervalCapacity
    >& schedule
) {
    using Traits = PathScheduleTraits<Schedule>;
    PreparedTerminalPathNode result{};
    if (scenario.step_count == scenario.central_step_count) {
        // Preserve the canonical price-only arithmetic for the central date.
        // Reconstructing T - prefix can differ by one FP32 ulp from the
        // calendar interval even when both represent the same contract.
        const std::uint32_t interval =
            Traits::kKind == PathScheduleKind::calendar
            ? schedule.prefix_observation_count
            : 0U;
        result.interval_years = schedule.central.interval_years[interval];
    } else {
        result.interval_years =
            scenario.maturity_years - schedule.prefix_years;
    }
    result.transition_count = Traits::kExactTransition
        ? 1U
        : scenario.step_count - schedule.prefix_step_count;
    return result;
}

template<typename Schedule>
inline void validate_path_schedule(
    const typename PathScheduleTraits<Schedule>::Calendar& calendar,
    pg::TimeConfiguration time
) {
    using Traits = PathScheduleTraits<Schedule>;
    if constexpr (Traits::kExactTransition) {
        simulation::validate_calendar(
            calendar,
            simulation::ExactTransitionTimeConfiguration{
                path_day_fraction(time)
            }
        );
    } else {
        simulation::validate_calendar(
            calendar,
            simulation::FixedStepTimeConfiguration{
                time.dt,
                time.simulation_steps_per_day,
            }
        );
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
