// Maturity-aligned regular schedules for early-exercise simulations.
#pragma once

#include "common/simulation/concepts.cuh"
#include "common/simulation/path_simulation.cuh"
#include "common/simulation/schedule.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::simulation {

struct MaturityAlignedExerciseCalendar {
    std::uint32_t maturity_days;
    std::uint32_t exercise_interval_days;
};

// Regular exercise dates beginning after one explicit initial stub.
struct RegularExerciseCalendar {
    std::uint32_t first_exercise_days;
    std::uint32_t exercise_interval_days;
    std::uint32_t exercise_count;
};

inline void validate_exercise_calendar(
    const RegularExerciseCalendar& calendar
) {
    if (calendar.first_exercise_days == 0U) {
        throw std::invalid_argument(
            "The first early-exercise date must be a positive day count."
        );
    }
    if (calendar.exercise_interval_days == 0U) {
        throw std::invalid_argument(
            "The early-exercise interval must be a positive day count."
        );
    }
    if (calendar.exercise_count < 2U) {
        throw std::invalid_argument(
            "An early-exercise calendar requires at least two dates."
        );
    }
}

inline void validate_exercise_calendar(
    const MaturityAlignedExerciseCalendar& calendar
) {
    if (calendar.maturity_days == 0U) {
        throw std::invalid_argument(
            "Early-exercise maturity must be a positive day count."
        );
    }
    if (calendar.exercise_interval_days == 0U
        || calendar.exercise_interval_days >= calendar.maturity_days) {
        throw std::invalid_argument(
            "Early-exercise interval must be positive and below maturity."
        );
    }
}

__host__ __device__ constexpr std::uint32_t maturity_aligned_exercise_count(
    const MaturityAlignedExerciseCalendar& calendar
) noexcept {
    return 1U + (calendar.maturity_days - 1U)
        / calendar.exercise_interval_days;
}

__host__ __device__ constexpr std::uint32_t
maturity_aligned_first_exercise_days(
    const MaturityAlignedExerciseCalendar& calendar
) noexcept {
    return calendar.maturity_days
        - (maturity_aligned_exercise_count(calendar) - 1U)
            * calendar.exercise_interval_days;
}

template<FixedStepDynamicsPolicy DynamicsPolicy>
struct FixedStepMaturityAlignedExerciseSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = FixedStepTimeConfiguration;
    using Calendar = MaturityAlignedExerciseCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t initial_transition_count;
        std::uint32_t transitions_per_exercise;
        std::uint32_t exercise_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        return {
            Dynamics::prepare_dynamics(parameters, time_configuration.dt),
            time_configuration.simulation_steps_per_day
                * maturity_aligned_first_exercise_days(calendar),
            time_configuration.simulation_steps_per_day
                * calendar.exercise_interval_days,
            maturity_aligned_exercise_count(calendar),
        };
    }

    __device__ __forceinline__ static std::uint32_t exercise_count(
        const PreparedSchedule& schedule
    ) {
        return schedule.exercise_count;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_fixed_step_stubbed_regular_schedule<Dynamics>(
            schedule.dynamics,
            schedule.initial_transition_count,
            schedule.transitions_per_exercise,
            schedule.exercise_count,
            key,
            path,
            handler
        );
    }
};

template<ExactTransitionDynamicsPolicy DynamicsPolicy>
struct ExactTransitionMaturityAlignedExerciseSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = ExactTransitionTimeConfiguration;
    using Calendar = MaturityAlignedExerciseCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition initial_transition;
        typename Dynamics::PreparedTransition regular_transition;
        std::uint32_t exercise_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        const typename Dynamics::PreparedModel model =
            Dynamics::prepare_model(parameters);
        return {
            model,
            Dynamics::prepare_transition(
                model,
                day_count_year_fraction(
                    maturity_aligned_first_exercise_days(calendar),
                    time_configuration
                )
            ),
            Dynamics::prepare_transition(
                model,
                day_count_year_fraction(
                    calendar.exercise_interval_days,
                    time_configuration
                )
            ),
            maturity_aligned_exercise_count(calendar),
        };
    }

    __device__ __forceinline__ static std::uint32_t exercise_count(
        const PreparedSchedule& schedule
    ) {
        return schedule.exercise_count;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_exact_transition_stubbed_regular_schedule<Dynamics>(
            schedule.model,
            schedule.initial_transition,
            schedule.regular_transition,
            schedule.exercise_count,
            key,
            path,
            handler
        );
    }
};

template<FixedStepDynamicsPolicy DynamicsPolicy>
struct FixedStepRegularExerciseSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = FixedStepTimeConfiguration;
    using Calendar = RegularExerciseCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t initial_transition_count;
        std::uint32_t transitions_per_exercise;
        std::uint32_t exercise_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        return {
            Dynamics::prepare_dynamics(parameters, time_configuration.dt),
            time_configuration.simulation_steps_per_day
                * calendar.first_exercise_days,
            time_configuration.simulation_steps_per_day
                * calendar.exercise_interval_days,
            calendar.exercise_count,
        };
    }

    __device__ __forceinline__ static std::uint32_t exercise_count(
        const PreparedSchedule& schedule
    ) {
        return schedule.exercise_count;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_fixed_step_stubbed_regular_schedule<Dynamics>(
            schedule.dynamics,
            schedule.initial_transition_count,
            schedule.transitions_per_exercise,
            schedule.exercise_count,
            key,
            path,
            handler
        );
    }
};

template<ExactTransitionDynamicsPolicy DynamicsPolicy>
struct ExactTransitionRegularExerciseSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = ExactTransitionTimeConfiguration;
    using Calendar = RegularExerciseCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition initial_transition;
        typename Dynamics::PreparedTransition regular_transition;
        std::uint32_t exercise_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        const typename Dynamics::PreparedModel model =
            Dynamics::prepare_model(parameters);
        return {
            model,
            Dynamics::prepare_transition(
                model,
                day_count_year_fraction(
                    calendar.first_exercise_days,
                    time_configuration
                )
            ),
            Dynamics::prepare_transition(
                model,
                day_count_year_fraction(
                    calendar.exercise_interval_days,
                    time_configuration
                )
            ),
            calendar.exercise_count,
        };
    }

    __device__ __forceinline__ static std::uint32_t exercise_count(
        const PreparedSchedule& schedule
    ) {
        return schedule.exercise_count;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_exact_transition_stubbed_regular_schedule<Dynamics>(
            schedule.model,
            schedule.initial_transition,
            schedule.regular_transition,
            schedule.exercise_count,
            key,
            path,
            handler
        );
    }
};

}  // namespace ai_factory::workbench::simulation
