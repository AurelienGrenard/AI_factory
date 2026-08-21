// Reusable CUDA interface for exact Normal-Inverse-Gaussian increments.
#pragma once

#include "common/philox.cuh"
#include "model/equity/normal_inverse_gaussian/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::normal_inverse_gaussian {

struct PreparedModel {
    float initial_log_spot;
    float drift_rate;
    float delta;
    float inverse_gamma;
    float beta;
};

struct PreparedTransition {
    float drift;
    float inverse_gaussian_mean;
    float inverse_gaussian_shape;
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
    const PreparedModel& model,
    const PreparedTransition& transition,
    float inverse_gaussian_increment,
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

__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
);

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

}  // namespace ai_factory::workbench::normal_inverse_gaussian
