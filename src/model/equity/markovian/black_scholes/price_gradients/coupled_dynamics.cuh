// Public Black-Scholes innovation adapter interface for multi-scenario terminal pricing.
#pragma once
#include "model/equity/markovian/black_scholes/dynamics.cuh"

namespace ai_factory::workbench::model::equity::black_scholes::price_gradients {
struct CoupledDynamics {
    using ModelParameters = black_scholes::ModelParameters;
    using State = black_scholes::State;
    using Prepared = black_scholes::PreparedDynamics;
    using RandomContext = DynamicsPolicy::RandomContext;
    struct Innovations { float normals[3]; };
    static constexpr bool kExactTerminal = true;
    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&);
    __device__ static void transition(const Prepared&, const Innovations&, const float*, State&);
    __device__ static float spot(const State&);
};
}  // namespace ai_factory::workbench::model::equity::black_scholes::price_gradients
