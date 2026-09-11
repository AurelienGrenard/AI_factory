// Public three-state SABR interface preserving normalized initial volatility under spot bumps.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"
#include "model/equity/markovian/sabr/dynamics.cuh"

namespace ai_factory::workbench::model::equity::sabr {

struct PriceDeltaDynamics {
    using ModelParameters = sabr::ModelParameters;
    using RandomContext = DynamicsPolicy::RandomContext;
    struct Parameters {
        ModelParameters model;
        float lower_spot;
        float upper_spot;
    };
    struct PreparedDynamics {
        sabr::PreparedDynamics central, lower, upper;
    };
    struct State { sabr::State central, lower, upper; };
    static constexpr bool kPartitionInvariantAdvance = true;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters, float dt
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& prepared
    );
    __device__ __forceinline__ static void advance(
        const PreparedDynamics& prepared, std::uint32_t step_count,
        RandomContext& random, State& state
    );
    template<unsigned int Scenario>
    __device__ __forceinline__ static
    ::ai_factory::workbench::equity::price_delta::SpotObservation observe(
        const State& state
    );
};

static_assert(simulation::FixedStepDynamicsPolicy<PriceDeltaDynamics>);

}  // namespace ai_factory::workbench::model::equity::sabr
