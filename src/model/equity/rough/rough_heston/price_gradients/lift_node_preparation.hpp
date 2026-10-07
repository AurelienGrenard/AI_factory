// Host preparation policy for cached rough_heston sensitivity nodes.
#pragma once

#include "common/volterra/price_gradients/prepared_lift_node_cache.hpp"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::equity::rough_heston::price_gradients {

template<std::size_t FactorCount>
struct LiftNodePreparation {
    using Model = ModelParameters;
    using Prepared = PreparedDynamics<FactorCount>;
    using Kernel = volterra::ExponentialKernel<FactorCount>;
    static_assert(sizeof(Model) == 9U * sizeof(float));

    static Prepared prepare(
        const Model& model,
        const Kernel& kernel,
        float horizon,
        float dt
    ) {
        return rough_heston::prepare_dynamics(model, kernel, horizon, dt);
    }
};

template<std::size_t FactorCount>
using PreparedLiftNodeCache =
    volterra::price_gradients::PreparedLiftNodeCache<
        FactorCount, LiftNodePreparation<FactorCount>
    >;

}  // namespace ai_factory::workbench::model::equity::rough_heston::price_gradients
