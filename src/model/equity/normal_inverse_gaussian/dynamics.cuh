// Reusable CUDA interface for exact Normal-Inverse-Gaussian increments.
#pragma once

#include "common/philox.cuh"
#include "model/equity/normal_inverse_gaussian/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::normal_inverse_gaussian {

// Coefficients prepared once per result row and fixed simulation time step.
struct NormalInverseGaussianPreparedParameters {
    float initial_log_spot;
    float inverse_gaussian_mean;
    float inverse_gaussian_shape;
    float beta;
    float drift_dt;
};

// Evolving log-spot private to one Monte Carlo path.
struct NormalInverseGaussianState {
    float log_spot;
};

// Terminal state and arithmetic mean observed from time zero to maturity.
struct NormalInverseGaussianMeanPathResult {
    NormalInverseGaussianState terminal_state;
    float arithmetic_mean;
};

// Terminal state and geometric mean observed from time zero to maturity.
struct NormalInverseGaussianGeometricMeanPathResult {
    NormalInverseGaussianState terminal_state;
    float geometric_mean;
};

// Path states observed at two requested times without global-memory storage.
struct NormalInverseGaussianTwoTimePathResult {
    NormalInverseGaussianState first_state;
    NormalInverseGaussianState terminal_state;
};

// Terminal state and maximum spot observed from time zero to maturity.
struct NormalInverseGaussianMaximumPathResult {
    NormalInverseGaussianState terminal_state;
    float maximum_spot;
};

// Precompute one exact increment law over the requested time interval.
__device__ __forceinline__ NormalInverseGaussianPreparedParameters
prepare_model(
    const NormalInverseGaussianModelParameters& parameters,
    float time_interval
);

// Precompute one monitored sub-step and its risk-neutral drift correction.
__device__ __forceinline__ NormalInverseGaussianPreparedParameters
prepare_model(
    const NormalInverseGaussianModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

// Construct the time-zero log-spot state.
__device__ __forceinline__ NormalInverseGaussianState initial_state(
    const NormalInverseGaussianPreparedParameters& model
);

// Apply one exact subordinated-Brownian increment to the log-spot.
__device__ __forceinline__ void one_step_transition(
    const NormalInverseGaussianPreparedParameters& model,
    float inverse_gaussian_increment,
    float brownian_normal,
    NormalInverseGaussianState& state
);

// Simulate one complete path and return its terminal state.
__device__ __forceinline__ NormalInverseGaussianState simulate_terminal_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
);

// Simulate one path and average its spots from time zero to maturity in FP64.
__device__ __forceinline__ NormalInverseGaussianMeanPathResult
simulate_mean_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate one path and average its log-spots before one final exponential.
__device__ __forceinline__ NormalInverseGaussianGeometricMeanPathResult
simulate_geometric_mean_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate two consecutive intervals and return both boundary states.
__device__ __forceinline__ NormalInverseGaussianTwoTimePathResult
simulate_at_two_times(
    const NormalInverseGaussianPreparedParameters& first_model,
    const NormalInverseGaussianPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
);

// Simulate one path and return its maximum monitored spot.
__device__ __forceinline__ NormalInverseGaussianMaximumPathResult
simulate_maximum_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Store exercise_count - 1 spots; the maturity state is only returned.
__device__ __forceinline__ NormalInverseGaussianState
simulate_on_regular_grid(
    const NormalInverseGaussianPreparedParameters& initial_stub_model,
    const NormalInverseGaussianPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
);

}  // namespace ai_factory::workbench::normal_inverse_gaussian
