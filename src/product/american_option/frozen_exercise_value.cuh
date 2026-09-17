// Shared FP32 payoff order for centered frozen-exercise sensitivities.
#pragma once

#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/option_side.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "common/philox.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<typename Schedule, OptionSide Side, typename Continuation>
__device__ __forceinline__ float simulate_frozen_exercise_path(
    const typename Schedule::PreparedSchedule& schedule,
    philox::PhiloxKey key,
    std::size_t state_offset,
    std::uint32_t regression_count,
    float strike,
    longstaff_schwartz::FrozenExerciseTrace* exercises,
    std::size_t path,
    std::size_t paths,
    const typename Continuation::StateView& states
) {
    auto writer = Continuation::make_writer(
        states,
        state_offset + path,
        paths,
        regression_count
    );
    const auto terminal = Schedule::simulate(schedule, key, path, writer);
    const float spot = Schedule::Dynamics::spot(terminal);
    exercises[path] = {regression_count, spot};
    return payoff::vanilla_option_payoff<Side>(spot, strike);
}

template<typename Continuation>
__device__ __forceinline__ void record_frozen_exercise(
    longstaff_schwartz::FrozenExerciseTrace* exercises,
    const typename Continuation::StateView& states,
    std::size_t observation,
    std::size_t path,
    std::uint32_t regression_count,
    std::uint32_t backward_level
) {
    exercises[path] = {
        regression_count - 1U - backward_level,
        Continuation::spot(states, observation),
    };
}

template<OptionSide Side>
__device__ __forceinline__ float centered_frozen_exercise_gradient(
    longstaff_schwartz::FrozenExerciseTrace exercise,
    float first_spot,
    float second_spot,
    float strike,
    float initial_discount,
    float exercise_discount,
    float represented_width
) {
    float first = payoff::vanilla_option_payoff<Side>(first_spot, strike);
    float second = payoff::vanilla_option_payoff<Side>(second_spot, strike);
    for (std::uint32_t date = 0U;
         date < exercise.observation;
         ++date) {
        first = exercise_discount * first;
        second = exercise_discount * second;
    }
    first = initial_discount * first;
    second = initial_discount * second;
    return (second - first) / represented_width;
}

}  // namespace ai_factory::workbench::product
