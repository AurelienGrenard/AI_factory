// Heston QE-M replay of two bumped paths through one shared Philox stream.
#pragma once

#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "model/equity/markovian/heston/price_gradients/coupled_dynamics.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::heston::price_gradients {

struct FrozenExerciseReplay {
    using Dynamics = CoupledDynamics;

    struct Prepared {
        Dynamics::Prepared first;
        Dynamics::Prepared second;
        float first_scale;
        float second_scale;
        bool reuse_first;
        bool reuse_second;
    };

    struct Spots {
        float first;
        float second;
    };

    template<typename Scenario>
    __device__ __forceinline__ static Prepared prepare(
        const Scenario&,
        const Scenario& first,
        const Scenario& second,
        float dt
    ) {
        auto first_model = first.model;
        auto second_model = second.model;
        first_model.spot = first.simulation_spot;
        second_model.spot = second.simulation_spot;
        return {
            Dynamics::prepare(first_model, dt),
            Dynamics::prepare(second_model, dt),
            first.spot_scale,
            second.spot_scale,
            first.reuse_central,
            second.reuse_central,
        };
    }

    __device__ __forceinline__ static Spots evaluate(
        const Prepared& prepared,
        longstaff_schwartz::FrozenExerciseTrace exercise,
        philox::PhiloxKey key,
        std::size_t path,
        std::uint32_t initial_transition_count,
        std::uint32_t transitions_per_exercise
    ) {
        if (prepared.reuse_first && prepared.reuse_second) {
            return {
                exercise.spot * prepared.first_scale,
                exercise.spot * prepared.second_scale,
            };
        }

        typename Dynamics::State first = Dynamics::initial(prepared.first);
        typename Dynamics::State second = Dynamics::initial(prepared.second);
        typename Dynamics::RandomContext random(
            key, static_cast<std::uint64_t>(path)
        );
        const std::uint64_t total =
            static_cast<std::uint64_t>(initial_transition_count)
            + static_cast<std::uint64_t>(exercise.observation)
                * transitions_per_exercise;
        for (std::uint64_t step = 0U; step < total; ++step) {
            const typename Dynamics::Innovations innovations =
                Dynamics::draw(random);
            Dynamics::transition(prepared.first, innovations, nullptr, first);
            Dynamics::transition(prepared.second, innovations, nullptr, second);
        }
        return {
            prepared.reuse_first
                ? exercise.spot * prepared.first_scale
                : Dynamics::spot(first) * prepared.first_scale,
            prepared.reuse_second
                ? exercise.spot * prepared.second_scale
                : Dynamics::spot(second) * prepared.second_scale,
        };
    }
};

}  // namespace ai_factory::workbench::model::equity::heston::price_gradients
