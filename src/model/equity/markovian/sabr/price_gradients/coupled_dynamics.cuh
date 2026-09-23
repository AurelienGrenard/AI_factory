// Public coupled-dynamics interface for SABR sensitivity nodes.
#pragma once

#include "model/equity/markovian/sabr/dynamics.cuh"

namespace ai_factory::workbench::model::equity::sabr::price_gradients {

struct CoupledDynamics {
    using ModelParameters = sabr::ModelParameters;
    using State = sabr::State;
    using Prepared = sabr::PreparedModel;
    using RandomContext = DynamicsPolicy::RandomContext;
    struct Innovations { float alpha_normal, residual_normal; };
    static constexpr bool kExactTerminal = false;
    static constexpr bool kDrawRequiresCentralPrepared = false;

    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&);
    __device__ static void transition(const Prepared&, const Innovations&, const float*, State&);
    __device__ static float spot(const State&);
};

}  // namespace ai_factory::workbench::model::equity::sabr::price_gradients
