// Reusable CUDA interface for Schobel-Zhu stochastic-volatility paths.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/schobel_zhu/parameters.hpp"

#include <cstdint>

namespace ai_factory::workbench::schobel_zhu {

struct PreparedModel {
    float initial_log_spot;
    float initial_volatility;
    float long_run_volatility;
    float exp_mean_reversion_dt;
    float ou_std;
    float endpoint_increment_correlation;
    float endpoint_increment_residual;
    float drift_dt;
    float sqrt_dt;
    float correlation;
    float correlation_residual;
};
using PreparedDynamics = PreparedModel;
struct State {
    float log_spot;
    float volatility;
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
    float ou_normal,
    float increment_residual_normal,
    float asset_residual_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = schobel_zhu::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = schobel_zhu::State;

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
};

static_assert(equity::LogSpotDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::schobel_zhu
