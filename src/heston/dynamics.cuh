// Reusable CUDA interface for Heston paths simulated with Andersen QE-M.
#pragma once

#include "common/philox.cuh"
#include "heston/parameters.hpp"

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

// Precompute row- and time-step-dependent QE-M coefficients once per block.
__device__ __forceinline__ HestonQeParameters prepare_model(
    const HestonModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

// Advance one path state by one Andersen QE-M time step.
__device__ __forceinline__ void one_step_qe_martingale_transition(
    const HestonQeParameters& model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    HestonState& state
);

// Simulate one complete path and return only its terminal spot.
__device__ __forceinline__ float simulate_terminal_spot(
    const HestonQeParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Store pre-maturity exercise states and return the terminal spot.
__device__ __forceinline__ float simulate_spot_variance_on_regular_grid(
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
