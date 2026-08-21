// Reusable CUDA interfaces for exact Vasicek simulations.
#pragma once

#include "common/philox.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::vasicek {

struct IntegralMoments {
    float state_loading;
    float mean_increment;
    float variance;
};

__device__ __forceinline__ float integral_state_loading(
    float mean_reversion,
    float delta
);

__device__ __forceinline__ float integral_variance(
    const ProcessParameters& parameters,
    float delta
);

__device__ __forceinline__ IntegralMoments integral_moments(
    const ProcessParameters& parameters,
    float delta
);

// Time-invariant coefficients of one Vasicek process.
struct PreparedModel {
    float mean_reversion;
    float long_term_mean;
    float volatility_squared;
};

// Coefficients required to advance the prepared model by one delta_t.
struct PreparedTransition {
    float decay;
    float mean_increment;
    float state_standard_deviation;
};

__device__ __forceinline__ PreparedModel prepare_model(
    const ProcessParameters& parameters
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

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float state_normal,
    float& state
);

__device__ __forceinline__ float simulate_terminal_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ float simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states
);

__device__ __forceinline__ float simulate_on_regular_grid(
    const PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states
);

namespace joint {

struct PreparedTransition {
    float decay;
    float state_mean_increment;
    float state_standard_deviation;
    float integral_state_loading;
    float integral_mean_increment;
    float integral_state_normal_loading;
    float integral_independent_standard_deviation;
};

struct State {
    float state;
    float state_integral;
};

__device__ __forceinline__ PreparedTransition prepare_transition(
    const vasicek::PreparedModel& model,
    float delta_t
);

__device__ __forceinline__ void prepare_calendar(
    const vasicek::PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
);

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float state_normal,
    float integral_normal,
    State& state
);

__device__ __forceinline__ State simulate_terminal_state(
    const vasicek::PreparedModel& model,
    const PreparedTransition& transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ State simulate_on_calendar(
    const vasicek::PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
);

__device__ __forceinline__ State simulate_on_regular_grid(
    const vasicek::PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
);

}  // namespace joint
}  // namespace ai_factory::workbench::model::vasicek
