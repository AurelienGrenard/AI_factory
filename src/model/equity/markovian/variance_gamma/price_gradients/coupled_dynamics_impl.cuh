// Included device definitions of the coupled exact Variance-Gamma increment adapter.
#pragma once

#include "model/equity/markovian/variance_gamma/price_gradients/coupled_dynamics.cuh"

#include "model/equity/markovian/variance_gamma/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::variance_gamma::price_gradients {

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters,
    float years
) {
    const auto model = variance_gamma::prepare_model(parameters);
    return {model, variance_gamma::prepare_transition(model, years)};
}

__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(
    const Prepared& prepared
) {
    return variance_gamma::initial_state(prepared.model);
}

__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(
    RandomContext& random,
    const Prepared& prepared
) {
    return variance_gamma::draw_transition_innovations(
        prepared.model, prepared.transition, random
    );
}

template<std::size_t NodeCapacity>
__device__ __forceinline__ void CoupledDynamics::draw_coupled(
    RandomContext& random,
    const Prepared (&prepared)[NodeCapacity],
    std::uint16_t node_count,
    Innovations (&innovations)[NodeCapacity]
) {
    const auto step = random.next_step();
    float central_gamma = 0.0f;
    for (std::uint16_t node = 0U; node < node_count; ++node) {
        const bool same_clock = node != 0U
            && prepared[node].model.nu == prepared[0U].model.nu
            && prepared[node].transition.gamma_shape
                == prepared[0U].transition.gamma_shape;
        if (same_clock) {
            innovations[node].gamma_increment = central_gamma;
            continue;
        }
        // Every node restarts the same address. Rejection counts may differ,
        // but cannot shift another node or the Brownian source.
        auto uniforms = random.source<
            variance_gamma::random_source::kGammaClock
        >(step);
        philox::NormalPairCache cache;
        const float gamma = philox::marsaglia_tsang_gamma(
            uniforms,
            cache,
            prepared[node].transition.gamma_shape,
            prepared[node].model.nu
        );
        innovations[node].gamma_increment = gamma;
        if (node == 0U) central_gamma = gamma;
    }

    auto brownian_uniforms = random.source<
        variance_gamma::random_source::kSubordinatedBrownian
    >(step);
    philox::NormalPairCache brownian_cache;
    const float brownian = philox::next_normal(
        brownian_uniforms, brownian_cache
    );
    for (std::uint16_t node = 0U; node < node_count; ++node) {
        innovations[node].brownian_normal = brownian;
    }
}

__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared,
    const Innovations& innovations,
    const float*,
    State& state
) {
    variance_gamma::one_step_transition(
        prepared.model,
        prepared.transition,
        innovations.gamma_increment,
        innovations.brownian_normal,
        state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return variance_gamma::DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::variance_gamma::price_gradients
