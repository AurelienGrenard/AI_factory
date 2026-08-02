// Reusable CUDA interface for Heston paths simulated with Andersen QE-M.
#pragma once

#include "common/philox.cuh"
#include "model/heston/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::heston {

// Coefficients prepared once per result row and reused by every path.
struct HestonQeParameters {
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
struct HestonState {
    float log_spot;
    float variance;
};

// Terminal state and arithmetic mean observed from time zero to maturity.
struct HestonMeanPathResult {
    HestonState terminal_state;
    float arithmetic_mean;
};

// Terminal state and maximum spot observed from time zero to maturity.
struct HestonMaximumPathResult {
    HestonState terminal_state;
    float maximum_spot;
};

// Precompute row- and time-step-dependent QE-M coefficients once per block.
__device__ __forceinline__ HestonQeParameters prepare_model(
    const HestonModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

// Construct the time-zero log-spot and variance state.
__device__ __forceinline__ HestonState initial_state(
    const HestonQeParameters& model
);

// Advance one path state by one Andersen QE-M time step.
__device__ __forceinline__ void one_step_transition(
    const HestonQeParameters& model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    HestonState& state
);

// Simulate one complete path and return its terminal state.
__device__ __forceinline__ HestonState simulate_terminal_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate one path and average its spots from time zero to maturity in FP64.
__device__ __forceinline__ HestonMeanPathResult simulate_mean_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate one path and return its maximum monitored spot.
__device__ __forceinline__ HestonMaximumPathResult simulate_maximum_state(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Store exercise_count - 1 states; the maturity state is only returned.
// Each date-major output needs (exercise_count - 1) * path_count values.
__device__ __forceinline__ HestonState simulate_on_regular_grid(
    const HestonQeParameters& initial_stub_model,
    const HestonQeParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_variances
);

}  // namespace ai_factory::workbench::heston
