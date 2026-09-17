// Public Merton common-innovation adapter for fixed-horizon terminal gradients.
#pragma once
#include "model/equity/markovian/merton/dynamics.cuh"

namespace ai_factory::workbench::model::equity::merton::price_gradients {
struct CoupledDynamics {
    using ModelParameters = merton::ModelParameters;
    using State = merton::State;
    using Prepared = merton::PreparedDynamics;
    using RandomContext = DynamicsPolicy::RandomContext;
    using Innovations = merton::TransitionInnovations;
    static constexpr bool kExactTerminal = true;
    static constexpr bool kDrawRequiresCentralPrepared = true;
    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&, const Prepared&);
    __device__ static void transition(const Prepared&, const Innovations&, const float*, State&);
    __device__ static float spot(const State&);
};
}  // namespace ai_factory::workbench::model::equity::merton::price_gradients
