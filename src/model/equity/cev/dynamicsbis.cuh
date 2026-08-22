// Experimental autonomous interface for absorbed Milstein CEV paths.
#pragma once

#include "common/equity/concepts.cuh"
#include "model/equity/cev/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::cev {

// Coefficients shared by every numerical transition of one pricing row.
struct PreparedModel {
    float initial_spot;
    float drift_dt;
    float diffusion_scale;
    float milstein_scale;
    float beta;
};

// For a fixed-step scheme, PreparedModel already contains the complete
// transition of duration delta_t.
using PreparedDynamics = PreparedModel;

// Mutable state of one path.
struct State {
    float spot;
};

// ===================== Model mathematical primitives =====================

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

// ======================== Compile-time policy ============================

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = cev::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = cev::State;

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

    __device__ __forceinline__ static float risk_free_rate(
        const Parameters& parameters
    );
};

static_assert(equity::EquityDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::cev
