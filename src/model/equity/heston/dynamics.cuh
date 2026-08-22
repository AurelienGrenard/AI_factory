// Reusable CUDA interface for Heston paths simulated with Andersen QE-M.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/heston/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::heston {

// Coefficients prepared once per result row and reused by every path.
struct PreparedModel {
    float initial_log_spot;
    float initial_variance;
    float theta;
    float exp_kdt;
    float variance_linear_scale;
    float variance_constant_scale;
    float drift_dt;
    float k0;
    float k1;
    float k2;
    float k3;
    float k4;
    float martingale_a;
};

// Evolving log-spot and variance private to one Monte Carlo path.
struct State {
    float log_spot;
    float variance;
};

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
    State& state
);

using PreparedDynamics = PreparedModel;

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = heston::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = heston::State;

    static constexpr bool kNativeLogSpot = true;

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

}  // namespace ai_factory::workbench::heston
