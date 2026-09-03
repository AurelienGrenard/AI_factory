// Compile-time capabilities for equity state observables.
#pragma once

#include "common/simulation/concepts.cuh"

#include <concepts>

namespace ai_factory::workbench::equity {

template<typename StateProvider>
concept SpotStatePolicy =
    simulation::StatePolicy<StateProvider>
    && requires(const typename StateProvider::State& state) {
        { StateProvider::spot(state) } -> std::same_as<float>;
    };

template<typename StateProvider>
concept LogSpotStatePolicy =
    SpotStatePolicy<StateProvider>
    && requires(const typename StateProvider::State& state) {
        { StateProvider::log_spot(state) } -> std::same_as<float>;
        { StateProvider::kNativeLogSpot } -> std::convertible_to<bool>;
    };

template<typename Dynamics>
concept SpotDynamicsPolicy =
    simulation::DynamicsPolicy<Dynamics> && SpotStatePolicy<Dynamics>;

template<typename Dynamics>
concept LogSpotDynamicsPolicy =
    simulation::DynamicsPolicy<Dynamics> && LogSpotStatePolicy<Dynamics>;

}  // namespace ai_factory::workbench::equity
