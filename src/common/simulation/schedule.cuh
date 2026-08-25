// Reusable terminal, regular, dense and calendar simulation schedules.
#pragma once

#include "common/check_cuda.cuh"
#include "common/simulation/concepts.cuh"
#include "common/simulation/path_simulation.cuh"
#include "common/time_configuration.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::simulation {

struct RegularCalendar {
    std::uint32_t observation_interval_days;
    std::uint32_t observation_count;
};

struct MaturityCalendar {
    std::uint32_t maturity_days;
};

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

// Convert a contractual day count while preserving the arithmetic used by
// the corresponding simulation family. In particular, fixed-step schemes
// multiply integer steps before the single FP32 conversion.
__device__ __forceinline__ float day_count_year_fraction(
    std::uint32_t day_count,
    const FixedStepTimeConfiguration& time_configuration
) {
    return static_cast<float>(
        time_configuration.simulation_steps_per_day * day_count
    ) * time_configuration.dt;
}

__device__ __forceinline__ float day_count_year_fraction(
    std::uint32_t day_count,
    const ExactTransitionTimeConfiguration& time_configuration
) {
    return time::year_fraction(day_count, time_configuration);
}

template<FixedStepDynamicsPolicy DynamicsPolicy>
struct FixedStepTerminalSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = FixedStepTimeConfiguration;
    using Calendar = MaturityCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transition_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        return {
            Dynamics::prepare_dynamics(parameters, time_configuration.dt),
            time_configuration.simulation_steps_per_day
                * calendar.maturity_days,
        };
    }

    __device__ __forceinline__ static typename Dynamics::State
    simulate_terminal(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        return simulate_fixed_step_terminal<Dynamics>(
            schedule.dynamics,
            schedule.transition_count,
            key,
            path
        );
    }

};

template<ExactTransitionDynamicsPolicy DynamicsPolicy>
struct ExactTransitionTerminalSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = ExactTransitionTimeConfiguration;
    using Calendar = MaturityCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition transition;
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
                    calendar.maturity_days,
                    time_configuration
                )
            ),
        };
    }

    __device__ __forceinline__ static typename Dynamics::State
    simulate_terminal(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        return simulate_exact_transition_terminal<Dynamics>(
            schedule.model,
            schedule.transition,
            key,
            path
        );
    }

};

template<FixedStepDynamicsPolicy DynamicsPolicy>
struct FixedStepRegularSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = FixedStepTimeConfiguration;
    using Calendar = RegularCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transitions_per_observation;
        std::uint32_t observation_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        const std::uint32_t transitions_per_observation =
            time_configuration.simulation_steps_per_day
                * calendar.observation_interval_days;
        return {
            Dynamics::prepare_dynamics(parameters, time_configuration.dt),
            transitions_per_observation,
            calendar.observation_count,
        };
    }

    __device__ __forceinline__ static std::uint32_t observation_count(
        const PreparedSchedule& schedule
    ) {
        return schedule.observation_count;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_fixed_step_regular_schedule<Dynamics>(
            schedule.dynamics,
            schedule.transitions_per_observation,
            schedule.observation_count,
            key,
            path,
            handler
        );
    }

};

template<ExactTransitionDynamicsPolicy DynamicsPolicy>
struct ExactTransitionRegularSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = ExactTransitionTimeConfiguration;
    using Calendar = RegularCalendar;

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition transition;
        std::uint32_t observation_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        const float observation_years = day_count_year_fraction(
            calendar.observation_interval_days,
            time_configuration
        );
        const typename Dynamics::PreparedModel model =
            Dynamics::prepare_model(parameters);
        return {
            model,
            Dynamics::prepare_transition(model, observation_years),
            calendar.observation_count,
        };
    }

    __device__ __forceinline__ static std::uint32_t observation_count(
        const PreparedSchedule& schedule
    ) {
        return schedule.observation_count;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_exact_transition_regular_schedule<Dynamics>(
            schedule.model,
            schedule.transition,
            schedule.observation_count,
            key,
            path,
            handler
        );
    }

};

// Observe every numerical step, including the initial state through the
// handler lifecycle. This is the dense-monitoring schedule used by Asian,
// lookback and discretely monitored barrier products.
template<FixedStepDynamicsPolicy DynamicsPolicy>
struct FixedStepDenseSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = FixedStepTimeConfiguration;
    using Calendar = MaturityCalendar;
    static constexpr bool kObservesEveryTransition = true;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transition_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        return {
            Dynamics::prepare_dynamics(parameters, time_configuration.dt),
            time_configuration.simulation_steps_per_day
                * calendar.maturity_days,
        };
    }

    __device__ __forceinline__ static std::uint32_t observation_count(
        const PreparedSchedule& schedule
    ) {
        return schedule.transition_count;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_fixed_step_dense_schedule<Dynamics>(
            schedule.dynamics,
            schedule.transition_count,
            key,
            path,
            handler
        );
    }

};

template<
    FixedStepDynamicsPolicy DynamicsPolicy,
    std::size_t ObservationCount
>
requires (ObservationCount > 0U)
struct FixedStepCalendarSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = FixedStepTimeConfiguration;
    static constexpr std::size_t kObservationCount = ObservationCount;

    struct Calendar {
        std::uint32_t interval_days[ObservationCount];
    };

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transition_counts[ObservationCount];
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        PreparedSchedule schedule{};
        schedule.dynamics = Dynamics::prepare_dynamics(
            parameters,
            time_configuration.dt
        );
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            schedule.transition_counts[observation] =
                time_configuration.simulation_steps_per_day
                * calendar.interval_days[observation];
        }
        return schedule;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_fixed_step_calendar<Dynamics>(
            schedule.dynamics,
            schedule.transition_counts,
            static_cast<std::uint32_t>(ObservationCount),
            key,
            path,
            handler
        );
    }

};

// A small compile-time calendar stores invariant exact-model coefficients once
// and one compact transition per contractual interval.
template<
    ExactTransitionDynamicsPolicy DynamicsPolicy,
    std::size_t ObservationCount
>
requires (ObservationCount > 0U)
struct ExactTransitionCalendarSchedule {
    using Dynamics = DynamicsPolicy;
    using TimeConfiguration = ExactTransitionTimeConfiguration;
    static constexpr std::size_t kObservationCount = ObservationCount;

    struct Calendar {
        std::uint32_t interval_days[ObservationCount];
    };

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition transitions[ObservationCount];
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Calendar& calendar,
        const TimeConfiguration& time_configuration
    ) {
        PreparedSchedule schedule{};
        schedule.model = Dynamics::prepare_model(parameters);
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            schedule.transitions[observation] =
                Dynamics::prepare_transition(
                    schedule.model,
                    day_count_year_fraction(
                        calendar.interval_days[observation],
                        time_configuration
                    )
                );
        }
        return schedule;
    }

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        return simulate_exact_transition_calendar<Dynamics>(
            schedule.model,
            schedule.transitions,
            static_cast<std::uint32_t>(ObservationCount),
            key,
            path,
            handler
        );
    }

};

}  // namespace ai_factory::workbench::simulation
