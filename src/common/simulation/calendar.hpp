// Host-safe simulation calendars and validation shared by pricing front ends.
#pragma once

#include "common/check_cuda.cuh"
#include "common/time_configuration.cuh"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::simulation {

struct RegularCalendar {
    std::uint32_t observation_interval_days;
    std::uint32_t observation_count;
};

struct StubbedRegularCalendar {
    std::uint32_t first_observation_day;
    std::uint32_t observation_interval_days;
    std::uint32_t observation_count;
};

struct MaturityCalendar {
    std::uint32_t maturity_days;
};

template<std::size_t ObservationCount>
requires (ObservationCount > 0U)
struct StaticCalendar {
    static constexpr std::size_t kObservationCount = ObservationCount;
    std::uint32_t interval_days[ObservationCount];
};

__host__ __device__ inline std::uint64_t calendar_maturity_days(
    const MaturityCalendar& calendar
) {
    return calendar.maturity_days;
}

// Contractual time immediately before the terminal event. Terminal-time
// sensitivities keep this prefix unchanged and move only the last event.
__host__ __device__ inline std::uint64_t calendar_terminal_prefix_days(
    const MaturityCalendar&
) {
    return 0U;
}

__host__ __device__ inline std::uint64_t calendar_maturity_days(
    const RegularCalendar& calendar
) {
    return static_cast<std::uint64_t>(calendar.observation_interval_days)
        * calendar.observation_count;
}

__host__ __device__ inline std::uint64_t calendar_terminal_prefix_days(
    const RegularCalendar& calendar
) {
    return static_cast<std::uint64_t>(calendar.observation_interval_days)
        * (calendar.observation_count - 1U);
}

__host__ __device__ inline std::uint64_t calendar_maturity_days(
    const StubbedRegularCalendar& calendar
) {
    return calendar.first_observation_day
        + static_cast<std::uint64_t>(calendar.observation_count - 1U)
            * calendar.observation_interval_days;
}

__host__ __device__ inline std::uint64_t calendar_terminal_prefix_days(
    const StubbedRegularCalendar& calendar
) {
    return calendar.observation_count == 1U
        ? 0U
        : calendar.first_observation_day
            + static_cast<std::uint64_t>(calendar.observation_count - 2U)
                * calendar.observation_interval_days;
}

template<std::size_t ObservationCount>
__host__ __device__ inline std::uint64_t calendar_maturity_days(
    const StaticCalendar<ObservationCount>& calendar
) {
    std::uint64_t maturity_days = 0U;
    #pragma unroll
    for (std::size_t observation = 0U;
         observation < ObservationCount;
         ++observation) {
        maturity_days += calendar.interval_days[observation];
    }
    return maturity_days;
}

template<std::size_t ObservationCount>
__host__ __device__ inline std::uint64_t calendar_terminal_prefix_days(
    const StaticCalendar<ObservationCount>& calendar
) {
    std::uint64_t prefix_days = 0U;
    #pragma unroll
    for (std::size_t observation = 0U;
         observation + 1U < ObservationCount;
         ++observation) {
        prefix_days += calendar.interval_days[observation];
    }
    return prefix_days;
}

struct FixedStepTimeConfiguration {
    float dt;
    std::uint32_t simulation_steps_per_day;
};

using ExactTransitionTimeConfiguration =
    time::DayFractionTimeConfiguration;

inline void validate_time_configuration(
    const FixedStepTimeConfiguration& time_configuration
) {
    validate_time_step(time_configuration.dt);
    validate_simulation_steps_per_day(
        time_configuration.simulation_steps_per_day
    );
}

inline void validate_time_configuration(
    const ExactTransitionTimeConfiguration& time_configuration
) {
    time::validate_time_configuration(time_configuration);
}

inline void validate_calendar(const MaturityCalendar& calendar) {
    if (calendar.maturity_days == 0U) {
        throw std::invalid_argument(
            "A terminal sample maturity must contain at least one day."
        );
    }
}

inline void validate_calendar(const RegularCalendar& calendar) {
    if (calendar.observation_interval_days == 0U
        || calendar.observation_count == 0U) {
        throw std::invalid_argument(
            "A regular calendar requires a positive interval and count."
        );
    }
}

inline void validate_calendar(const StubbedRegularCalendar& calendar) {
    if (calendar.first_observation_day == 0U
        || calendar.observation_interval_days == 0U
        || calendar.observation_count == 0U) {
        throw std::invalid_argument(
            "A stubbed regular calendar requires positive days and count."
        );
    }
}

template<typename Calendar>
requires requires(const Calendar& calendar) {
    { Calendar::kObservationCount } -> std::convertible_to<std::size_t>;
    calendar.interval_days[0U];
}
inline void validate_calendar(const Calendar& calendar) {
    static_assert(Calendar::kObservationCount > 0U);
    for (std::size_t observation = 0U;
         observation < Calendar::kObservationCount;
         ++observation) {
        if (calendar.interval_days[observation] == 0U) {
            throw std::invalid_argument(
                "An irregular calendar requires positive day intervals."
            );
        }
    }
}

// Validate every integer-to-transition conversion on the host before a CUDA
// launch. Fixed-step schedules store each interval count in uint32_t, so the
// checked arithmetic must use a wider type and must preserve a finite FP32
// year fraction before the device prepares the same value.
inline std::uint32_t checked_fixed_step_transition_count(
    std::uint32_t day_count,
    const FixedStepTimeConfiguration& time_configuration
) {
    ::ai_factory::workbench::simulation::validate_time_configuration(
        time_configuration
    );
    if (day_count == 0U) {
        throw std::invalid_argument(
            "A fixed-step interval must contain at least one day."
        );
    }
    const std::uint64_t transition_count =
        static_cast<std::uint64_t>(
            time_configuration.simulation_steps_per_day
        ) * day_count;
    if (transition_count
        > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "simulation_steps_per_day * day_count exceeds uint32_t."
        );
    }
    const float year_fraction = static_cast<float>(transition_count)
        * time_configuration.dt;
    if (!std::isfinite(year_fraction)) {
        throw std::overflow_error(
            "The fixed-step interval has a non-finite year fraction."
        );
    }
    return static_cast<std::uint32_t>(transition_count);
}

template<typename Calendar, typename TimeConfiguration>
inline void validate_calendar_maturity_year_fraction(
    const Calendar& calendar,
    const TimeConfiguration& time_configuration
) {
    float day_fraction = 0.0f;
    if constexpr (std::same_as<
            TimeConfiguration,
            FixedStepTimeConfiguration
        >) {
        day_fraction = static_cast<float>(
            time_configuration.simulation_steps_per_day
        ) * time_configuration.dt;
    } else {
        day_fraction = time_configuration.day_fraction;
    }
    const float maturity_years =
        static_cast<float>(calendar_maturity_days(calendar)) * day_fraction;
    if (!std::isfinite(day_fraction) || !std::isfinite(maturity_years)) {
        throw std::overflow_error(
            "The simulation calendar has a non-finite year fraction."
        );
    }
}

template<typename Calendar>
inline void validate_calendar(
    const Calendar& calendar,
    const FixedStepTimeConfiguration& time_configuration
) {
    validate_calendar(calendar);
    if constexpr (std::same_as<Calendar, MaturityCalendar>) {
        checked_fixed_step_transition_count(
            calendar.maturity_days,
            time_configuration
        );
    } else if constexpr (std::same_as<Calendar, RegularCalendar>) {
        checked_fixed_step_transition_count(
            calendar.observation_interval_days,
            time_configuration
        );
    } else if constexpr (std::same_as<Calendar, StubbedRegularCalendar>) {
        checked_fixed_step_transition_count(
            calendar.first_observation_day,
            time_configuration
        );
        checked_fixed_step_transition_count(
            calendar.observation_interval_days,
            time_configuration
        );
    } else {
        for (std::size_t observation = 0U;
             observation < Calendar::kObservationCount;
             ++observation) {
            checked_fixed_step_transition_count(
                calendar.interval_days[observation],
                time_configuration
            );
        }
    }
    validate_calendar_maturity_year_fraction(calendar, time_configuration);
}

template<typename Calendar>
inline void validate_calendar(
    const Calendar& calendar,
    const ExactTransitionTimeConfiguration& time_configuration
) {
    ::ai_factory::workbench::simulation::validate_time_configuration(
        time_configuration
    );
    validate_calendar(calendar);
    validate_calendar_maturity_year_fraction(calendar, time_configuration);
}

}  // namespace ai_factory::workbench::simulation
