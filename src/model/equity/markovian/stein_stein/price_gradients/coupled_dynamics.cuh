// Public Stein--Stein common-innovation adapter interface for sensitivity nodes.
#pragma once

#include "model/equity/markovian/stein_stein/dynamics.cuh"

namespace ai_factory::workbench::model::equity::stein_stein::price_gradients {

struct CoupledDynamics {
    using ModelParameters = stein_stein::ModelParameters;
    using State = stein_stein::State;
    using Prepared = stein_stein::PreparedModel;
    using RandomContext = stein_stein::DynamicsPolicy::RandomContext;
    struct Innovations {
        float endpoint_normal;
        float increment_residual_normal;
        float asset_residual_normal;
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

}  // namespace ai_factory::workbench::model::equity::stein_stein::price_gradients
