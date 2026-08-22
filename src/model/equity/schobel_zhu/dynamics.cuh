// Reusable CUDA interface for Schobel-Zhu stochastic-volatility paths.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/schobel_zhu/parameters.hpp"

#include <cstddef>
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

struct MeanPathResult {
    float arithmetic_mean;
};

struct GeometricMeanPathResult {
    float geometric_mean;
};

struct MaximumPathResult {
    float maximum_spot;
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

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

__device__ __forceinline__ MeanPathResult simulate_mean_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

__device__ __forceinline__ GeometricMeanPathResult
simulate_geometric_mean_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

__device__ __forceinline__ MaximumPathResult simulate_maximum_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

// Each output pointer addresses this path's first pre-terminal observation.
__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_volatilities
);

// Each output pointer addresses this path's first pre-terminal observation.
__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    const std::uint32_t* __restrict__ steps_between_observations,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_volatilities
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
    __device__ __forceinline__ static float risk_free_rate(
        const Parameters& parameters
    );
};

static_assert(equity::EquityDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::schobel_zhu
