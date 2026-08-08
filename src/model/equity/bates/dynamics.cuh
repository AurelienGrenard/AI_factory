// Reusable CUDA interface for Bates paths simulated with Andersen QE-M.
#pragma once

#include "common/philox.cuh"
#include "model/equity/bates/dataset.hpp"
#include "model/equity/heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::bates {

// Coefficients prepared once per result row and reused by every path.
struct BatesQeParameters {
    heston::HestonQeParameters heston;
    float poisson_mean;
    float poisson_zero_probability;
    float jump_log_mean;
    float jump_log_volatility;
    float jump_compensator;
};

// Evolving log-spot and variance private to one Monte Carlo path.
using BatesState = heston::HestonState;

// Terminal state and arithmetic mean observed from time zero to maturity.
struct BatesMeanPathResult {
    BatesState terminal_state;
    float arithmetic_mean;
};

// Terminal state and geometric mean observed from time zero to maturity.
struct BatesGeometricMeanPathResult {
    BatesState terminal_state;
    float geometric_mean;
};

// Path states observed at two requested times without global-memory storage.
struct BatesTwoTimePathResult {
    BatesState first_state;
    BatesState terminal_state;
};

// Terminal state and maximum spot observed from time zero to maturity.
struct BatesMaximumPathResult {
    BatesState terminal_state;
    float maximum_spot;
};

// Precompute row- and time-step-dependent QE-M coefficients once per block.
__device__ __forceinline__ BatesQeParameters prepare_model(
    const BatesModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

// Construct the time-zero log-spot and variance state.
__device__ __forceinline__ BatesState initial_state(
    const BatesQeParameters& model
);

// Advance one path state by one Andersen QE-M time step.
__device__ __forceinline__ void one_step_transition(
    const BatesQeParameters& model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    std::uint32_t jump_count,
    float jump_normal,
    BatesState& state
);

// Simulate one complete path and return its terminal state.
__device__ __forceinline__ BatesState simulate_terminal_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate one path and average its spots from time zero to maturity in FP64.
__device__ __forceinline__ BatesMeanPathResult simulate_mean_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate one path and average its log-spots before one final exponential.
__device__ __forceinline__ BatesGeometricMeanPathResult
simulate_geometric_mean_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate two consecutive intervals and return both boundary states.
__device__ __forceinline__ BatesTwoTimePathResult simulate_at_two_times(
    const BatesQeParameters& first_model,
    const BatesQeParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t first_num_steps,
    std::size_t second_num_steps
);

// Simulate one path and return its maximum monitored spot.
__device__ __forceinline__ BatesMaximumPathResult simulate_maximum_state(
    const BatesQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Store exercise_count - 1 states; the maturity state is only returned.
// Each date-major output needs (exercise_count - 1) * path_count values.
__device__ __forceinline__ BatesState simulate_on_regular_grid(
    const BatesQeParameters& initial_stub_model,
    const BatesQeParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_variances
);

}  // namespace ai_factory::workbench::bates
