// Public contract for the exact G2++ common-innovation sensitivity adapter.
#pragma once

#include "model/fixed_income/g2_plus_plus/dynamics.cuh"

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients {

struct CoupledDynamics {
    using ModelParameters = g2_plus_plus::ModelParameters;
    using Prepared = g2::joint::PreparedDynamics;
    using State = g2::joint::State;
    using RandomContext = philox::NormalRandomContext;
    struct Innovations { float x, y, integral; };

    static constexpr bool kExactTerminal = true;
    static constexpr bool kDrawRequiresCentralPrepared = false;

    __device__ static Prepared prepare(const ModelParameters&, float horizon);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&);
    __device__ static void transition(
        const Prepared&, const Innovations&, const float*, State&
    );
};

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients
