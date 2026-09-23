// Public common-normal adapter for canonical absorbed Milstein CEV transitions.
#pragma once
#include "model/equity/markovian/cev/dynamics.cuh"

namespace ai_factory::workbench::model::equity::cev::price_gradients {
struct CoupledDynamics {
    using ModelParameters = cev::ModelParameters;
    using State = cev::State;
    using Prepared = cev::PreparedModel;
    using RandomContext = DynamicsPolicy::RandomContext;
    struct Innovations { float normal; };
    static constexpr bool kExactTerminal = false;
    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&);
    __device__ static void transition(const Prepared&, const Innovations&, const float*, State&);
    __device__ static float spot(const State&);
};
}  // namespace ai_factory::workbench::model::equity::cev::price_gradients
