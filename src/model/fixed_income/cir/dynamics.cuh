// Reusable CUDA interfaces for exact CIR state simulations.
#pragma once

#include "common/philox.cuh"
#include "model/fixed_income/cir/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::cir {

// Time-invariant coefficients of the noncentral chi-square law.
struct PreparedModel {
    float mean_reversion;
    float degrees_of_freedom;
    float scale_rate;
};

// Coefficients required to advance the prepared model by one delta_t.
struct PreparedTransition {
    float decay;
    float scale;
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
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
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

}  // namespace ai_factory::workbench::model::cir
