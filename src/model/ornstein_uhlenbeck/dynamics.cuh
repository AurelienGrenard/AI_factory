// Reusable CUDA interface for an exact Ornstein-Uhlenbeck process.
#pragma once

#include "common/philox.cuh"
#include "model/ornstein_uhlenbeck/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Exact joint-transition coefficients reused at every equal time step.
struct OrnsteinUhlenbeckExactParameters {
    float decay;
    float factor_standard_deviation;
    float integral_factor_loading;
    float integral_factor_normal_loading;
    float integral_independent_standard_deviation;
};

// Gaussian factor and its accumulated integral private to one path.
struct OrnsteinUhlenbeckState {
    float factor;
    float integrated_factor;
};

// Precompute the exact factor and factor-integral transition coefficients.
__device__ __forceinline__ OrnsteinUhlenbeckExactParameters prepare_model(
    const OrnsteinUhlenbeckDynamicsParameters& parameters,
    float maturity,
    std::size_t num_steps
);

// Construct the deterministic time-zero factor and integral state.
__device__ __forceinline__ OrnsteinUhlenbeckState initial_state(
    float initial_factor
);

// Advance the factor and its integral exactly over one time step.
__device__ __forceinline__ void one_step_transition(
    const OrnsteinUhlenbeckExactParameters& model,
    float factor_normal,
    float integral_normal,
    OrnsteinUhlenbeckState& state
);

// Simulate one complete path and return its terminal factor and integral.
__device__ __forceinline__ OrnsteinUhlenbeckState simulate_terminal_state(
    const OrnsteinUhlenbeckExactParameters& model,
    float initial_factor,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Store pre-maturity factor states and return the terminal state.
__device__ __forceinline__ OrnsteinUhlenbeckState simulate_on_regular_grid(
    const OrnsteinUhlenbeckExactParameters& initial_stub_model,
    const OrnsteinUhlenbeckExactParameters& regular_model,
    float initial_factor,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_factors,
    float* __restrict__ observed_integrated_factors
);

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
