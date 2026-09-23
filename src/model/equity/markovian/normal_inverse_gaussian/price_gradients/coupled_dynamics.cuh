// Public NIG common-random-number adapter interface for exact increments.
#pragma once

#include "model/equity/markovian/normal_inverse_gaussian/dynamics.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian::price_gradients {

struct CoupledDynamics {
    using ModelParameters = normal_inverse_gaussian::ModelParameters;
    using State = normal_inverse_gaussian::State;
    using RandomContext = DynamicsPolicy::RandomContext;
    using Innovations = normal_inverse_gaussian::TransitionInnovations;

    struct Prepared {
        normal_inverse_gaussian::PreparedModel model;
        normal_inverse_gaussian::PreparedTransition transition;
    };

    static constexpr bool kExactTerminal = true;
    static constexpr bool kDrawRequiresCentralPrepared = true;

    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&, const Prepared&);
    template<std::size_t NodeCapacity>
    __device__ static void draw_coupled(
        RandomContext&,
        const Prepared (&)[NodeCapacity],
        std::uint8_t node_count,
        Innovations (&)[NodeCapacity]
    );
    __device__ static void transition(
        const Prepared&, const Innovations&, const float*, State&
    );
    __device__ static float spot(const State&);
};

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian::price_gradients
