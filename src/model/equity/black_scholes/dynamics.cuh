// Reusable CUDA interface for exact Black-Scholes transitions.
#pragma once

#include "common/philox.cuh"
#include "model/equity/black_scholes/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::black_scholes {

// Quantities computed once for one model and reused for every interval.
struct PreparedModel {
    float initial_log_spot;
    float drift_rate;
    float volatility;
};

// Exact Black-Scholes increment coefficients over one fixed interval.
struct PreparedTransition {
    float drift;
    float standard_deviation;
};

struct State {
    float log_spot;
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

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
);

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
);

__device__ __forceinline__ void prepare_calendar(
    const PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
);

__device__ __forceinline__ State initial_state(
    const PreparedModel& model
);

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float brownian_normal,
    State& state
);

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ MeanPathResult simulate_mean_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
);

__device__ __forceinline__ GeometricMeanPathResult
simulate_geometric_mean_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
);

__device__ __forceinline__ MaximumPathResult simulate_maximum_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
);

// Each output pointer addresses this path's first pre-terminal observation.
__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
);

// Each output pointer addresses this path's first pre-terminal observation.
__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
);

}  // namespace ai_factory::workbench::black_scholes
