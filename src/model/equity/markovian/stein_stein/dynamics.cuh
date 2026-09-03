// Public state and fixed-step dynamics-policy declarations for Stein--Stein.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/markovian/stein_stein/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::stein_stein {

struct State {
    float log_spot;
    float volatility;
};

struct PreparedModel {
    float initial_log_spot;
    float initial_volatility;
    float volatility_decay;
    float volatility_standard_deviation;
    float endpoint_increment_correlation;
    float endpoint_increment_residual;
    float drift_dt;
    float sqrt_dt;
    float rho;
    float correlation_residual;
};

using PreparedDynamics = PreparedModel;

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = stein_stein::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = stein_stein::State;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kPartitionInvariantAdvance = true;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters,
        float delta_t
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    );
    __device__ __forceinline__ static void simulate_one_step(
        const PreparedDynamics& dynamics,
        RandomContext& random,
        State& state
    );
    __device__ __forceinline__ static void advance(
        const PreparedDynamics& dynamics,
        std::uint32_t step_count,
        RandomContext& random,
        State& state
    );
    __device__ __forceinline__ static float spot(const State& state);
    __device__ __forceinline__ static float log_spot(const State& state);
};

static_assert(::ai_factory::workbench::equity::LogSpotDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::model::equity::stein_stein
