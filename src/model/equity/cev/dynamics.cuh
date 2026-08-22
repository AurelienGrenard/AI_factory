// Reusable CUDA interface for absorbed Milstein CEV paths.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/cev/parameters.hpp"

#include <cstdint>

namespace ai_factory::workbench::cev {

struct PreparedModel {
    float initial_spot;
    float drift_dt;
    float diffusion_scale;
    float milstein_scale;
    float beta;
};
using PreparedDynamics = PreparedModel;
struct State {
    float spot;
};

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
