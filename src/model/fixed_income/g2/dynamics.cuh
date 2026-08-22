// Reusable CUDA interfaces for exact correlated G2 simulations.
#pragma once

#include "common/philox.cuh"
#include "model/fixed_income/g2/parameters.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::g2 {

struct IntegralMoments {
    float state_x_loading;
    float state_y_loading;
    float variance;
};

__device__ __forceinline__ IntegralMoments integral_moments(
    const ProcessParameters& parameters,
    float delta
);

// Time-invariant coefficients of the correlated two-factor process.
struct PreparedModel {
    ProcessParameters process;
};

struct PreparedTransition {
    float decay_x;
    float state_x_standard_deviation;
    float decay_y;
    float state_y_x_normal_loading;
    float state_y_independent_standard_deviation;
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
    float state_x_normal,
    float state_y_normal,
    State& state
);

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    State initial_state,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y
);

__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y
);

namespace joint {

struct PreparedTransition {
    float decay_x;
    float state_x_standard_deviation;
    float decay_y;
    float state_y_x_normal_loading;
    float state_y_independent_standard_deviation;
    float integral_state_x_loading;
    float integral_state_y_loading;
    float integral_x_normal_loading;
    float integral_y_normal_loading;
    float integral_independent_standard_deviation;
};

struct State {
    g2::State state;
    float state_integral;
};

__device__ __forceinline__ PreparedTransition prepare_transition(
    const g2::PreparedModel& model,
    float delta_t
);

__device__ __forceinline__ void prepare_calendar(
    const g2::PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
);

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float state_x_normal,
    float state_y_normal,
    float integral_normal,
    State& state
);

__device__ __forceinline__ State simulate_terminal_state(
    const g2::PreparedModel& model,
    const PreparedTransition& transition,
    g2::State initial_state,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ State simulate_on_calendar(
    const g2::PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    g2::State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y,
    float* __restrict__ observed_integrated_states
);

__device__ __forceinline__ State simulate_on_regular_grid(
    const g2::PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    g2::State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y,
    float* __restrict__ observed_integrated_states
);

}  // namespace joint
}  // namespace ai_factory::workbench::model::g2
