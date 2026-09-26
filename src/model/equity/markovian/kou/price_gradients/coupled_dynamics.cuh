// Public Kou event-coupling adapter for direct-transition sensitivities.
#pragma once

#include "model/equity/markovian/kou/dynamics.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::kou::price_gradients {

struct CoupledDynamics {
    using ModelParameters = kou::ModelParameters;
    using State = kou::State;
    using RandomContext = DynamicsPolicy::RandomContext;
    using Innovations = kou::TransitionInnovations;

    struct Prepared {
        kou::PreparedModel model;
        kou::PreparedTransition transition;
        float horizon;
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

}  // namespace ai_factory::workbench::model::equity::kou::price_gradients
