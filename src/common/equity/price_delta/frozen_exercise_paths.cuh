// Spot reconstruction at a recorded exercise: scale a spot or replay coupled paths.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"
#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/philox.cuh"
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::equity::price_delta {

using FrozenExercise = longstaff_schwartz::FrozenExerciseTrace;
using InitialExerciseDecision =
    longstaff_schwartz::InitialExerciseDecision;

struct BumpedSpots {
    float lower;
    float upper;
};

struct MultiplicativeFrozenExercise {
    struct Prepared { float lower_scale; float upper_scale; };

    template<typename Model, typename Product, typename Time>
    __device__ __forceinline__ static Prepared prepare(const Model&, const Product&,
                                       const Time&, SpotBump bump) {
        return {bump.lower / bump.central, bump.upper / bump.central};
    }

    __device__ __forceinline__ static BumpedSpots evaluate(
        const Prepared& prepared, FrozenExercise exercise,
        philox::PhiloxKey, std::size_t
    ) {
        return {exercise.spot * prepared.lower_scale,
                exercise.spot * prepared.upper_scale};
    }
};

template<typename ReplaySchedule, typename PathPolicy>
struct CoupledFrozenExercise {
    using Dynamics = typename ReplaySchedule::Dynamics;
    static_assert(std::same_as<Dynamics, typename PathPolicy::Dynamics>);
    struct Prepared {
        typename ReplaySchedule::PreparedSchedule schedule;
        typename PathPolicy::Prepared path;
    };

    template<typename Model, typename Product, typename Time>
    __device__ __forceinline__ static Prepared prepare(const Model& model, const Product& product,
                                       const Time& time, SpotBump bump) {
        return {ReplaySchedule::prepare(PathPolicy::parameters(model, bump),
                    {product.maturity_days, product.exercise_interval_days}, time),
                PathPolicy::prepare(bump)};
    }

    struct StopAtExercise {
        std::uint32_t exercise;
        __device__ __forceinline__ bool on_initial_state(const typename Dynamics::State&) { return true; }
        __device__ __forceinline__ bool on_observation(std::uint32_t observation, const typename Dynamics::State&) {
            return observation < exercise;
        }
    };

    __device__ __forceinline__ static BumpedSpots evaluate(
        const Prepared& prepared, FrozenExercise exercise,
        philox::PhiloxKey key, std::size_t path
    ) {
        StopAtExercise stop{exercise.observation};
        const auto state = ReplaySchedule::simulate(prepared.schedule, key, path, stop);
        return {PathPolicy::template observe<1U>(prepared.path, state).spot,
                PathPolicy::template observe<2U>(prepared.path, state).spot};
    }
};

}  // namespace ai_factory::workbench::equity::price_delta
