// Experimental autonomous interface for exact Merton jump diffusion paths.
#pragma once

#include "common/equity/concepts.cuh"
#include "model/equity/merton/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::merton {

// Coefficients invariant with respect to the transition duration.
struct PreparedModel {
    float initial_log_spot;
    float drift_rate;
    float volatility;
    float jump_intensity;
    float jump_log_mean;
    float jump_log_volatility;
};

// Coefficients specific to one exact interval delta_t.
struct PreparedTransition {
    float drift;
    float diffusion_standard_deviation;
    float poisson_mean;
    float zero_jump_probability;
};

static_assert(sizeof(PreparedTransition) == 4U * sizeof(float));

// Compact homogeneous dynamics used by generic path algorithms.
struct PreparedDynamics {
    PreparedModel model;
    PreparedTransition transition;
};

// Mutable state of one path.
struct State {
    float log_spot;
};

// ===================== Model mathematical primitives =====================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
);

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
);

__device__ __forceinline__ State initial_state(
    const PreparedModel& model
);

__device__ __forceinline__ void one_step_transition(
    const PreparedModel& model,
    const PreparedTransition& transition,
    std::uint32_t jump_count,
    float diffusion_normal,
    float jump_normal,
    State& state
);

// ======================== Compile-time policy ============================

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = merton::PreparedDynamics;
    using PreparedModel = merton::PreparedModel;
    using PreparedTransition = merton::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = merton::State;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters,
        float delta_t
    );

    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters
    );

    __device__ __forceinline__ static PreparedTransition prepare_transition(
        const PreparedModel& model,
        float delta_t
    );

    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    );

    __device__ __forceinline__ static State initial_state(
        const PreparedModel& model
    );

    __device__ __forceinline__ static void simulate_one_step(
        const PreparedDynamics& dynamics,
        RandomContext& random,
        State& state
    );

    __device__ __forceinline__ static void advance(
        const PreparedDynamics& dynamics,
        std::uint32_t step_count,
        RandomContext& random,
        State& state
    );

    __device__ __forceinline__ static void simulate_one_step(
        const PreparedModel& model,
        const PreparedTransition& transition,
        RandomContext& random,
        State& state
    );

    __device__ __forceinline__ static float spot(const State& state);

    __device__ __forceinline__ static float log_spot(const State& state);

    __device__ __forceinline__ static float risk_free_rate(
        const Parameters& parameters
    );
};

static_assert(equity::EquityDynamicsPolicy<DynamicsPolicy>);
static_assert(equity::ExactTransitionDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::merton
