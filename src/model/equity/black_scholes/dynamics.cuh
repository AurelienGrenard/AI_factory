// Reusable CUDA interface for exact Black-Scholes transitions.
#pragma once

#include "common/philox.cuh"
#include "model/equity/black_scholes/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::black_scholes {

// Coefficients of one exact log-price transition over a fixed interval.
struct BlackScholesPreparedParameters {
    float initial_log_spot;
    float drift;
    float standard_deviation;
};

struct BlackScholesState {
    float log_spot;
};

struct BlackScholesMeanPathResult {
    float arithmetic_mean;
};

struct BlackScholesGeometricMeanPathResult {
    float geometric_mean;
};

struct BlackScholesTwoTimePathResult {
    float first_spot;
    float terminal_spot;
};

struct BlackScholesMaximumPathResult {
    float maximum_spot;
};

// ======================== Common equity dynamics =========================

__device__ __forceinline__ BlackScholesPreparedParameters prepare_model(
    const BlackScholesModelParameters& parameters,
    float time_interval
);

__device__ __forceinline__ BlackScholesPreparedParameters prepare_model(
    const BlackScholesModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

__device__ __forceinline__ BlackScholesState initial_state(
    const BlackScholesPreparedParameters& model
);

__device__ __forceinline__ void one_step_transition(
    const BlackScholesPreparedParameters& model,
    float brownian_normal,
    BlackScholesState& state
);

__device__ __forceinline__ BlackScholesState simulate_terminal_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ BlackScholesMeanPathResult simulate_mean_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ BlackScholesGeometricMeanPathResult
simulate_geometric_mean_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ BlackScholesTwoTimePathResult simulate_at_two_times(
    const BlackScholesPreparedParameters& first_model,
    const BlackScholesPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ BlackScholesMaximumPathResult simulate_maximum_state(
    const BlackScholesPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ BlackScholesState simulate_on_regular_grid(
    const BlackScholesPreparedParameters& initial_stub_model,
    const BlackScholesPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
);

}  // namespace ai_factory::workbench::black_scholes
