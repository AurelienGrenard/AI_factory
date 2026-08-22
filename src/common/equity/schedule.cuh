// Reusable dense, regular and calendar schedules for equity dynamics.
#pragma once

#include "common/check_cuda.cuh"
#include "common/equity/concepts.cuh"
#include "common/equity/path_simulation.cuh"
#include "common/equity/pricing_inputs.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::equity {

struct RegularScheduleDefinition {
    std::uint32_t observation_interval_days;
    std::uint32_t observation_count;
};

struct DenseScheduleDefinition {
    std::uint32_t maturity_days;
};

struct FixedStepConfiguration {
    float dt;
    std::uint32_t simulation_steps_per_day;
};

struct ExactTransitionConfiguration {
    float day_fraction;
};

static_assert(std::is_trivially_copyable_v<RegularScheduleDefinition>);
static_assert(std::is_trivially_copyable_v<DenseScheduleDefinition>);
static_assert(std::is_trivially_copyable_v<FixedStepConfiguration>);
static_assert(std::is_trivially_copyable_v<ExactTransitionConfiguration>);

// Convert a contractual day count while preserving the arithmetic used by
// the corresponding simulation family. In particular, fixed-step schemes
// multiply integer steps before the single FP32 conversion.
__device__ __forceinline__ float day_count_year_fraction(
    std::uint32_t day_count,
    const FixedStepConfiguration& configuration
) {
    return static_cast<float>(
        configuration.simulation_steps_per_day * day_count
    ) * configuration.dt;
}

__device__ __forceinline__ float day_count_year_fraction(
    std::uint32_t day_count,
    const ExactTransitionConfiguration& configuration
) {
    return static_cast<float>(day_count) * configuration.day_fraction;
}

template<EquityDynamicsPolicy DynamicsPolicy>
struct FixedStepTerminalSchedule {
    using Dynamics = DynamicsPolicy;
    using Configuration = FixedStepConfiguration;
    using DeviceInputs = EmptyDeviceInputs;
    using Definition = DenseScheduleDefinition;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transition_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Definition& definition,
        const Configuration& configuration,
        const DeviceInputs&
    ) {
        return {
            Dynamics::prepare_dynamics(parameters, configuration.dt),
            configuration.simulation_steps_per_day
                * definition.maturity_days,
        };
    }

    __device__ __forceinline__ static float interval_year_fraction(
        const Definition&,
        std::uint32_t,
        const Configuration& configuration
    ) {
        return configuration.dt;
    }

    __device__ __forceinline__ static float total_year_fraction(
        const Definition& definition,
        const Configuration& configuration
    ) {
        return static_cast<float>(
            configuration.simulation_steps_per_day
                * definition.maturity_days
        ) * configuration.dt;
    }

    __device__ __forceinline__ static std::uint32_t observation_count(
        const PreparedSchedule&
    ) {
        return 1U;
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

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        typename Dynamics::State state = Dynamics::initial_state(
            schedule.dynamics
        );
        if (!handler.on_initial_state(state)) return state;
        typename Dynamics::RandomContext random(
            key,
            static_cast<std::uint64_t>(path)
        );
        Dynamics::advance(
            schedule.dynamics,
            schedule.transition_count,
            random,
            state
        );
        handler.on_observation(0U, state);
        return state;
    }

    static void validate_configuration(
        const Configuration& configuration,
        const DeviceInputs&,
        std::size_t monte_carlo_paths_per_price
    ) {
        validate_monte_carlo_parameters(
            monte_carlo_paths_per_price,
            configuration.dt
        );
        validate_simulation_steps_per_day(
            configuration.simulation_steps_per_day
        );
    }
};

template<ExactTransitionDynamicsPolicy DynamicsPolicy>
struct ExactTransitionTerminalSchedule {
    using Dynamics = DynamicsPolicy;
    using Configuration = ExactTransitionConfiguration;
    using DeviceInputs = EmptyDeviceInputs;
    using Definition = DenseScheduleDefinition;

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition transition;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Definition& definition,
        const Configuration& configuration,
        const DeviceInputs&
    ) {
        const typename Dynamics::PreparedModel model =
            Dynamics::prepare_model(parameters);
        return {
            model,
            Dynamics::prepare_transition(
                model,
                total_year_fraction(definition, configuration)
            ),
        };
    }

    __device__ __forceinline__ static float interval_year_fraction(
        const Definition& definition,
        std::uint32_t,
        const Configuration& configuration
    ) {
        return total_year_fraction(definition, configuration);
    }

    __device__ __forceinline__ static float total_year_fraction(
        const Definition& definition,
        const Configuration& configuration
    ) {
        return static_cast<float>(definition.maturity_days)
            * configuration.day_fraction;
    }

    __device__ __forceinline__ static std::uint32_t observation_count(
        const PreparedSchedule&
    ) {
        return 1U;
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

    template<ObservationHandlerFor<Dynamics> Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule,
        philox::PhiloxKey key,
        std::size_t path,
        Handler& handler
    ) {
        typename Dynamics::State state = Dynamics::initial_state(
            schedule.model
        );
        if (!handler.on_initial_state(state)) return state;
        typename Dynamics::RandomContext random(
            key,
            static_cast<std::uint64_t>(path)
        );
        Dynamics::simulate_one_step(
            schedule.model,
            schedule.transition,
            random,
            state
        );
        handler.on_observation(0U, state);
        return state;
    }

    static void validate_configuration(
        const Configuration& configuration,
        const DeviceInputs&,
        std::size_t monte_carlo_paths_per_price
    ) {
        validate_monte_carlo_path_count(monte_carlo_paths_per_price);
        validate_day_fraction(configuration.day_fraction);
    }
};

template<EquityDynamicsPolicy DynamicsPolicy>
struct FixedStepRegularSchedule {
    using Dynamics = DynamicsPolicy;
    using Configuration = FixedStepConfiguration;
    using DeviceInputs = EmptyDeviceInputs;
    using Definition = RegularScheduleDefinition;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transitions_per_observation;
        std::uint32_t observation_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Definition& definition,
        const Configuration& configuration,
        const DeviceInputs&
    ) {
        const std::uint32_t transitions_per_observation =
            configuration.simulation_steps_per_day
                * definition.observation_interval_days;
        return {
            Dynamics::prepare_dynamics(parameters, configuration.dt),
            transitions_per_observation,
            definition.observation_count,
        };
    }

    __device__ __forceinline__ static float interval_year_fraction(
        const Definition& definition,
        std::uint32_t,
        const Configuration& configuration
    ) {
        const std::uint32_t transitions_per_observation =
            configuration.simulation_steps_per_day
                * definition.observation_interval_days;
        return static_cast<float>(transitions_per_observation)
            * configuration.dt;
    }

    __device__ __forceinline__ static float total_year_fraction(
        const Definition& definition,
        const Configuration& configuration
    ) {
        return static_cast<float>(definition.observation_count)
            * interval_year_fraction(definition, 0U, configuration);
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

    static void validate_configuration(
        const Configuration& configuration,
        const DeviceInputs&,
        std::size_t monte_carlo_paths_per_price
    ) {
        validate_monte_carlo_parameters(
            monte_carlo_paths_per_price,
            configuration.dt
        );
        validate_simulation_steps_per_day(
            configuration.simulation_steps_per_day
        );
    }
};

template<ExactTransitionDynamicsPolicy DynamicsPolicy>
struct ExactTransitionRegularSchedule {
    using Dynamics = DynamicsPolicy;
    using Configuration = ExactTransitionConfiguration;
    using DeviceInputs = EmptyDeviceInputs;
    using Definition = RegularScheduleDefinition;

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition transition;
        std::uint32_t observation_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Definition& definition,
        const Configuration& configuration,
        const DeviceInputs&
    ) {
        const float observation_years = interval_year_fraction(
            definition,
            0U,
            configuration
        );
        const typename Dynamics::PreparedModel model =
            Dynamics::prepare_model(parameters);
        return {
            model,
            Dynamics::prepare_transition(model, observation_years),
            definition.observation_count,
        };
    }

    __device__ __forceinline__ static float interval_year_fraction(
        const Definition& definition,
        std::uint32_t,
        const Configuration& configuration
    ) {
        return static_cast<float>(definition.observation_interval_days)
            * configuration.day_fraction;
    }

    __device__ __forceinline__ static float total_year_fraction(
        const Definition& definition,
        const Configuration& configuration
    ) {
        return static_cast<float>(definition.observation_count)
            * interval_year_fraction(definition, 0U, configuration);
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

    static void validate_configuration(
        const Configuration& configuration,
        const DeviceInputs&,
        std::size_t monte_carlo_paths_per_price
    ) {
        validate_monte_carlo_path_count(monte_carlo_paths_per_price);
        validate_day_fraction(configuration.day_fraction);
    }
};

// Observe every numerical step, including the initial state through the
// handler lifecycle. This is the dense-monitoring schedule used by Asian,
// lookback and discretely monitored barrier products.
template<EquityDynamicsPolicy DynamicsPolicy>
struct FixedStepDenseSchedule {
    using Dynamics = DynamicsPolicy;
    using Configuration = FixedStepConfiguration;
    using DeviceInputs = EmptyDeviceInputs;
    using Definition = DenseScheduleDefinition;
    static constexpr bool kObservesEveryTransition = true;

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transition_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Definition& definition,
        const Configuration& configuration,
        const DeviceInputs&
    ) {
        return {
            Dynamics::prepare_dynamics(parameters, configuration.dt),
            configuration.simulation_steps_per_day
                * definition.maturity_days,
        };
    }

    __device__ __forceinline__ static float interval_year_fraction(
        const Definition&,
        std::uint32_t,
        const Configuration& configuration
    ) {
        return configuration.dt;
    }

    __device__ __forceinline__ static float total_year_fraction(
        const Definition& definition,
        const Configuration& configuration
    ) {
        return static_cast<float>(
            configuration.simulation_steps_per_day
                * definition.maturity_days
        ) * configuration.dt;
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
        return simulate_fixed_step_regular_schedule<Dynamics>(
            schedule.dynamics,
            1U,
            schedule.transition_count,
            key,
            path,
            handler
        );
    }

    static void validate_configuration(
        const Configuration& configuration,
        const DeviceInputs&,
        std::size_t monte_carlo_paths_per_price
    ) {
        validate_monte_carlo_parameters(
            monte_carlo_paths_per_price,
            configuration.dt
        );
        validate_simulation_steps_per_day(
            configuration.simulation_steps_per_day
        );
    }
};

template<
    EquityDynamicsPolicy DynamicsPolicy,
    std::size_t ObservationCount
>
requires (ObservationCount > 0U)
struct FixedStepCalendarSchedule {
    using Dynamics = DynamicsPolicy;
    using Configuration = FixedStepConfiguration;
    using DeviceInputs = EmptyDeviceInputs;
    static constexpr std::size_t kObservationCount = ObservationCount;

    struct Definition {
        std::uint32_t interval_days[ObservationCount];
    };

    struct PreparedSchedule {
        typename Dynamics::PreparedDynamics dynamics;
        std::uint32_t transition_counts[ObservationCount];
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Definition& definition,
        const Configuration& configuration,
        const DeviceInputs&
    ) {
        PreparedSchedule schedule{};
        schedule.dynamics = Dynamics::prepare_dynamics(
            parameters,
            configuration.dt
        );
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            schedule.transition_counts[observation] =
                configuration.simulation_steps_per_day
                * definition.interval_days[observation];
        }
        return schedule;
    }

    __device__ __forceinline__ static float interval_year_fraction(
        const Definition& definition,
        std::uint32_t observation,
        const Configuration& configuration
    ) {
        return static_cast<float>(
            configuration.simulation_steps_per_day
                * definition.interval_days[observation]
        ) * configuration.dt;
    }

    __device__ __forceinline__ static float total_year_fraction(
        const Definition& definition,
        const Configuration& configuration
    ) {
        std::uint32_t total_days = 0U;
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            total_days += definition.interval_days[observation];
        }
        return static_cast<float>(
            configuration.simulation_steps_per_day * total_days
        ) * configuration.dt;
    }

    __device__ __forceinline__ static std::uint32_t observation_count(
        const PreparedSchedule&
    ) {
        return static_cast<std::uint32_t>(ObservationCount);
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

    static void validate_configuration(
        const Configuration& configuration,
        const DeviceInputs&,
        std::size_t monte_carlo_paths_per_price
    ) {
        validate_monte_carlo_parameters(
            monte_carlo_paths_per_price,
            configuration.dt
        );
        validate_simulation_steps_per_day(
            configuration.simulation_steps_per_day
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
    using Configuration = ExactTransitionConfiguration;
    using DeviceInputs = EmptyDeviceInputs;
    static constexpr std::size_t kObservationCount = ObservationCount;

    struct Definition {
        std::uint32_t interval_days[ObservationCount];
    };

    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::PreparedTransition transitions[ObservationCount];
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& parameters,
        const Definition& definition,
        const Configuration& configuration,
        const DeviceInputs&
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
                    interval_year_fraction(
                        definition,
                        static_cast<std::uint32_t>(observation),
                        configuration
                    )
                );
        }
        return schedule;
    }

    __device__ __forceinline__ static float interval_year_fraction(
        const Definition& definition,
        std::uint32_t observation,
        const Configuration& configuration
    ) {
        return static_cast<float>(definition.interval_days[observation])
            * configuration.day_fraction;
    }

    __device__ __forceinline__ static float total_year_fraction(
        const Definition& definition,
        const Configuration& configuration
    ) {
        std::uint32_t total_days = 0U;
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < ObservationCount;
             ++observation) {
            total_days += definition.interval_days[observation];
        }
        return static_cast<float>(total_days) * configuration.day_fraction;
    }

    __device__ __forceinline__ static std::uint32_t observation_count(
        const PreparedSchedule&
    ) {
        return static_cast<std::uint32_t>(ObservationCount);
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

    static void validate_configuration(
        const Configuration& configuration,
        const DeviceInputs&,
        std::size_t monte_carlo_paths_per_price
    ) {
        validate_monte_carlo_path_count(monte_carlo_paths_per_price);
        validate_day_fraction(configuration.day_fraction);
    }
};

}  // namespace ai_factory::workbench::equity
