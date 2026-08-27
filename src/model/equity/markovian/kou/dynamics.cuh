// Reusable CUDA interface for exact Kou jump-diffusion increments.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/markovian/kou/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::kou {

struct State {
    float log_spot;
};

struct PreparedModel {
    float initial_log_spot;
    float drift_rate;
    float volatility;
    float jump_intensity;
    float up_probability;
    float inverse_positive_jump_rate;
    float inverse_negative_jump_rate;
};

struct PreparedTransition {
    float drift;
    float diffusion_standard_deviation;
    float poisson_mean;
    float zero_jump_probability;
};

// The transition is intentionally four FP32 scalars: one compact interval row.
static_assert(sizeof(PreparedTransition) == 4U * sizeof(float));

struct PreparedDynamics {
    PreparedModel model;
    PreparedTransition transition;
};

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
);

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
);

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
);

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& prepared_transition,
    float diffusion_normal,
    float jump_log_sum,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = kou::PreparedDynamics;
    using PreparedModel = kou::PreparedModel;
    using PreparedTransition = kou::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = kou::State;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kPartitionInvariantAdvance = true;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters, float delta_t
    );
    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters
    );
    __device__ __forceinline__ static PreparedTransition prepare_transition(
        const PreparedModel& prepared_model, float delta_t
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedModel& prepared_model
    );
    __device__ __forceinline__ static void simulate_one_step(
        const PreparedDynamics& dynamics, RandomContext& random, State& state
    );
    __device__ __forceinline__ static void advance(
        const PreparedDynamics& dynamics, std::uint32_t step_count,
        RandomContext& random, State& state
    );
    __device__ __forceinline__ static void simulate_one_step(
        const PreparedModel& prepared_model, const PreparedTransition& prepared_transition,
        RandomContext& random, State& state
    );
    __device__ __forceinline__ static float spot(const State& state);
    __device__ __forceinline__ static float log_spot(const State& state);
};

static_assert(::ai_factory::workbench::equity::LogSpotDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::ExactTransitionDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::model::equity::kou
