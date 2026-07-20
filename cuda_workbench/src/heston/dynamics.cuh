// Reusable CUDA interface for Heston paths simulated with Andersen QE-M.
#pragma once

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
    std::uint64_t seed;
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
    std::size_t num_steps,
    std::uint64_t seed
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
    std::size_t path,
    std::size_t num_steps
);

}  // namespace ai_factory::workbench::heston
