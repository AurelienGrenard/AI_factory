// Public Schobel--Zhu common-innovation adapter interface for sensitivity nodes.
#pragma once

#include "model/equity/markovian/schobel_zhu/dynamics.cuh"

namespace ai_factory::workbench::model::equity::schobel_zhu::price_gradients {

struct CoupledDynamics {
    using ModelParameters = schobel_zhu::ModelParameters;
    using State = schobel_zhu::State;
    using Prepared = schobel_zhu::PreparedModel;
    using RandomContext = schobel_zhu::DynamicsPolicy::RandomContext;
    struct Innovations {
        float ou_normal;
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

}  // namespace ai_factory::workbench::model::equity::schobel_zhu::price_gradients
