// Reusable CUDA interface for absorbed Milstein CEV paths.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/markovian/cev/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::cev {

struct State {
    float spot;
};

struct PreparedModel {
    float initial_spot;
    float drift_dt;
    float diffusion_scale;
    float milstein_scale;
    float beta;
};

using PreparedDynamics = PreparedModel;

// ======================== Common equity dynamics =========================

// Prepare the coefficients defining one transition of duration delta_t
// under the supplied model parameters.
__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float delta_t
);

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
);

__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model,
    float normal,
    State& state
);

// Stateless compile-time adapter consumed by generic equity path algorithms.
struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = cev::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = cev::State;

    static constexpr bool kNativeLogSpot = false;
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

}  // namespace ai_factory::workbench::model::equity::cev
