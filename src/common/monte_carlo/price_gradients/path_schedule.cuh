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
