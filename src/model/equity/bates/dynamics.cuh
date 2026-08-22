// Reusable CUDA interface for Bates paths simulated with Andersen QE-M.
#pragma once

#include "common/philox.cuh"
#include "model/equity/bates/parameters.hpp"
#include "model/equity/heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::bates {

// Coefficients prepared once per result row and reused by every path.
struct PreparedModel {
    heston::PreparedModel heston;
    float poisson_mean;
    float poisson_zero_probability;
    float jump_log_mean;
    float jump_log_volatility;
    float jump_compensator;
};

// Evolving log-spot and variance private to one Monte Carlo path.
using State = heston::State;

// Arithmetic mean observed from time zero to maturity.
struct MeanPathResult {
    float arithmetic_mean;
};

// Geometric mean observed from time zero to maturity.
struct GeometricMeanPathResult {
    float geometric_mean;
};

// Maximum spot observed from time zero to maturity.
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

// Simulate one complete path and return its terminal state.
__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

// Simulate one path and average its spots from time zero to maturity in FP64.
__device__ __forceinline__ MeanPathResult simulate_mean_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

// Simulate one path and average its log-spots before one final exponential.
__device__ __forceinline__ GeometricMeanPathResult
simulate_geometric_mean_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

// Simulate one path and return its maximum monitored spot.
__device__ __forceinline__ MaximumPathResult simulate_maximum_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

// Output pointers address this path's first pre-terminal observation.
// Store observation_count - 1 states; the terminal state is only returned.
__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_variances
);

// Output pointers address this path's first pre-terminal observation.
__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    const std::uint32_t* __restrict__ steps_between_observations,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_variances
);

}  // namespace ai_factory::workbench::bates
