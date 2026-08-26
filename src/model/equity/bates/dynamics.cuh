// Reusable CUDA interface for Bates paths simulated with Andersen QE-M.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/bates/parameters.hpp"
#include "model/equity/heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::bates {

// Bates and Heston share the same path state exactly.
using State = heston::State;

// Coefficients prepared once per result row and reused by every path.
struct PreparedModel {
    heston::PreparedModel heston;
    float poisson_mean;
    float zero_jump_probability;
    float jump_log_mean;
    float jump_log_volatility;
    float jump_compensator;
};

using PreparedDynamics = PreparedModel;

// ======================== Common equity dynamics =========================

// Prepare the coefficients defining one transition of duration delta_t
// under the supplied model parameters.
__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float delta_t
);

// Construct the time-zero log-spot and variance state.
__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
);

// Advance one path state by one Andersen QE-M time step.
__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    std::uint32_t jump_count,
    float jump_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = bates::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = bates::State;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kPartitionInvariantAdvance = false;

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

}  // namespace ai_factory::workbench::model::equity::bates
