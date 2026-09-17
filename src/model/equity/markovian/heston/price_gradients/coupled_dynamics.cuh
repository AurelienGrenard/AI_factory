// Public Heston QE-M innovation adapter interface for multi-scenario terminal pricing.
#pragma once
#include "model/equity/markovian/heston/dynamics.cuh"

namespace ai_factory::workbench::model::equity::heston::price_gradients {
struct CoupledDynamics {
    using ModelParameters = heston::ModelParameters;
    using State = heston::State;
    using Prepared = heston::PreparedModel;
    using RandomContext = DynamicsPolicy::RandomContext;
    struct Innovations { float variance_normal, variance_uniform, stock_normal; };
    static constexpr bool kExactTerminal = false;
    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&);
    __device__ static void transition(const Prepared&, const Innovations&, const float*, State&);
    __device__ static float spot(const State&);
};
}  // namespace ai_factory::workbench::model::equity::heston::price_gradients
