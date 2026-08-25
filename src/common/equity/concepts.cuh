// Compile-time capabilities for equity state observables.
#pragma once

#include "common/simulation/concepts.cuh"

#include <concepts>

namespace ai_factory::workbench::equity {

template<typename Dynamics>
concept SpotDynamicsPolicy =
    simulation::DynamicsPolicy<Dynamics>
    && requires(const typename Dynamics::State& state) {
        { Dynamics::spot(state) } -> std::same_as<float>;
    };

template<typename Dynamics>
concept LogSpotDynamicsPolicy =
    SpotDynamicsPolicy<Dynamics>
    && requires(const typename Dynamics::State& state) {
        { Dynamics::log_spot(state) } -> std::same_as<float>;
        { Dynamics::kNativeLogSpot } -> std::convertible_to<bool>;
    };

}  // namespace ai_factory::workbench::equity
