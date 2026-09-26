// Public Variance-Gamma common-random-number adapter interface for exact increments.
#pragma once

#include "model/equity/markovian/variance_gamma/dynamics.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::variance_gamma::price_gradients {

struct CoupledDynamics {
    using ModelParameters = variance_gamma::ModelParameters;
    using State = variance_gamma::State;
    using RandomContext = DynamicsPolicy::RandomContext;
    using Innovations = variance_gamma::TransitionInnovations;

    struct Prepared {
        variance_gamma::PreparedModel model;
        variance_gamma::PreparedTransition transition;
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
        std::uint16_t node_count,
        Innovations (&)[NodeCapacity]
    );
    __device__ static void transition(
        const Prepared&, const Innovations&, const float*, State&
    );
    __device__ static float spot(const State&);
};

}  // namespace ai_factory::workbench::model::equity::variance_gamma::price_gradients
