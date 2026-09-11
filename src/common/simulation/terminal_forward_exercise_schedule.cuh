// Exact, time-inhomogeneous exercise transitions under the last-exercise bond.
// Variable-length coefficients live in LSM observation storage, never in a path.
#pragma once

#include "common/simulation/early_exercise_schedule.cuh"

namespace ai_factory::workbench::simulation {

// A terminal-forward transition depends on both the interval and the remaining
// numeraire tenor; it must not masquerade as a time-homogeneous exact transition.
template<typename Dynamics>
concept TerminalForwardDynamicsPolicy = DynamicsPolicy<Dynamics>
    && requires(const typename Dynamics::Parameters& parameters,
                const typename Dynamics::PreparedModel& model,
                const typename Dynamics::PreparedTransition& transition,
                typename Dynamics::RandomContext& random,
                typename Dynamics::State& state, float interval, float remaining) {
        { Dynamics::prepare_model(parameters) } -> std::same_as<typename Dynamics::PreparedModel>;
        { Dynamics::initial_state(model) } -> std::same_as<typename Dynamics::State>;
        { Dynamics::prepare_transition(parameters, interval, remaining) }
            -> std::same_as<typename Dynamics::PreparedTransition>;
        { Dynamics::simulate_one_step(model, transition, random, state) } -> std::same_as<void>;
    };

template<TerminalForwardDynamicsPolicy ForwardDynamics>
struct TerminalForwardRegularExerciseSchedule {
    using Dynamics = ForwardDynamics;
    using Calendar = RegularExerciseCalendar;
    using TimeConfiguration = ExactTransitionTimeConfiguration;
    struct Observation {
        typename Dynamics::PreparedTransition transition;
        float log_numeraire_a;
        float numeraire_b;
    };
    struct PreparedSchedule {
        typename Dynamics::PreparedModel model;
        typename Dynamics::Parameters parameters;
        const Observation* observations;
        float initial_log_numeraire;
        std::uint32_t exercise_count;
    };

    __device__ __forceinline__ static PreparedSchedule prepare(
        const typename Dynamics::Parameters& model, const Calendar& calendar,
        const TimeConfiguration&
    ) {
        return {Dynamics::prepare_model(model), model, nullptr, 0.0f, calendar.exercise_count};
    }

    __device__ __forceinline__ static std::uint32_t exercise_count(
        const PreparedSchedule& schedule
    ) { return schedule.exercise_count; }

    template<typename Handler>
    __device__ __forceinline__ static typename Dynamics::State simulate(
        const PreparedSchedule& schedule, philox::PhiloxKey key,
        std::size_t path, Handler& handler
    ) {
        typename Dynamics::RandomContext random(key, path);
        auto state = Dynamics::initial_state(schedule.model);
        if (!handler.on_initial_state(state)) return state;
        for (std::uint32_t exercise = 0U; exercise < schedule.exercise_count; ++exercise) {
            Dynamics::simulate_one_step(
                schedule.model, schedule.observations[exercise].transition, random, state
            );
            if (!handler.on_observation(exercise, state)) break;
        }
        return state;
    }
};

}  // namespace ai_factory::workbench::simulation
