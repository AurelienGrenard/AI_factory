// Reusable CUDA interface for exact Variance-Gamma increments.
#pragma once

#include "common/philox.cuh"
#include "model/equity/variance_gamma/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::variance_gamma {

// Coefficients prepared once per result row and fixed simulation time step.
struct VarianceGammaPreparedParameters {
    float initial_log_spot;
    float gamma_shape;
    float gamma_scale;
    float theta;
    float sigma;
    float drift_dt;
};

// Evolving log-spot private to one Monte Carlo path.
struct VarianceGammaState {
    float log_spot;
};

// Arithmetic mean observed from time zero to maturity.
struct VarianceGammaMeanPathResult {
    float arithmetic_mean;
};

// Geometric mean observed from time zero to maturity.
struct VarianceGammaGeometricMeanPathResult {
    float geometric_mean;
};

// Spot values observed at two requested times without retaining extra state.
struct VarianceGammaTwoTimePathResult {
    float first_spot;
    float terminal_spot;
};

// Maximum spot observed from time zero to maturity.
struct VarianceGammaMaximumPathResult {
    float maximum_spot;
};

// Precompute one exact increment law over the requested time interval.
__device__ __forceinline__ VarianceGammaPreparedParameters prepare_model(
    const VarianceGammaModelParameters& parameters,
    float time_interval
);

// Precompute one monitored sub-step and its risk-neutral drift correction.
__device__ __forceinline__ VarianceGammaPreparedParameters prepare_model(
    const VarianceGammaModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

// Construct the time-zero log-spot state.
__device__ __forceinline__ VarianceGammaState initial_state(
    const VarianceGammaPreparedParameters& model
);

// Apply one exact subordinated-Brownian increment to the log-spot.
__device__ __forceinline__ void one_step_transition(
    const VarianceGammaPreparedParameters& model,
    float gamma_increment,
    float brownian_normal,
    VarianceGammaState& state
);

// Simulate one complete path and return its terminal state.
__device__ __forceinline__ VarianceGammaState simulate_terminal_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
);

// Simulate one path and average its spots from time zero to maturity in FP64.
__device__ __forceinline__ VarianceGammaMeanPathResult simulate_mean_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate one path and average its log-spots before one final exponential.
__device__ __forceinline__ VarianceGammaGeometricMeanPathResult
simulate_geometric_mean_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Simulate two consecutive intervals and return the two boundary spots only.
__device__ __forceinline__ VarianceGammaTwoTimePathResult
simulate_at_two_times(
    const VarianceGammaPreparedParameters& first_model,
    const VarianceGammaPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
);

// Simulate one path and return its maximum monitored spot.
__device__ __forceinline__ VarianceGammaMaximumPathResult
simulate_maximum_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Store exercise_count - 1 spots; the maturity state is only returned.
__device__ __forceinline__ VarianceGammaState simulate_on_regular_grid(
    const VarianceGammaPreparedParameters& initial_stub_model,
    const VarianceGammaPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
);

}  // namespace ai_factory::workbench::variance_gamma
