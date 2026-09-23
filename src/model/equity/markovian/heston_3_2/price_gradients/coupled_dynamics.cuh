// Public Heston 3/2 common-innovation adapter interface for sensitivity nodes.
#pragma once

#include "model/equity/markovian/heston_3_2/dynamics.cuh"

namespace ai_factory::workbench::model::equity::heston_3_2::price_gradients {

struct CoupledDynamics {
    using ModelParameters = heston_3_2::ModelParameters;
    using State = heston_3_2::State;
    using Prepared = heston_3_2::PreparedModel;
    using RandomContext = heston_3_2::DynamicsPolicy::RandomContext;
    struct Innovations {
        float variance_normal;
        float residual_normal;
    };
    static constexpr bool kExactTerminal = false;
    static constexpr bool kDrawRequiresCentralPrepared = false;

    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&);
    __device__ static void transition(
        const Prepared&, const Innovations&, const float*, State&
    );
    __device__ static float spot(const State&);
};

}  // namespace ai_factory::workbench::model::equity::heston_3_2::price_gradients
