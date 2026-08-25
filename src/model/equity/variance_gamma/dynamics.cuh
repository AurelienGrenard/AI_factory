// Reusable CUDA interface for exact Variance-Gamma increments.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/variance_gamma/parameters.hpp"

#include <cstdint>

namespace ai_factory::workbench::variance_gamma {

struct PreparedModel {
    float initial_log_spot;
    float drift_rate;
    float inverse_nu;
    float nu;
    float theta;
    float sigma;
};

struct PreparedTransition {
    float drift;
    float gamma_shape;
};

struct State {
    float log_spot;
};

// ======================== Common equity dynamics =========================

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
    float gamma_increment,
    float brownian_normal,
    State& state
);

struct PreparedDynamics {
    PreparedModel model;
    PreparedTransition transition;
};

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = variance_gamma::PreparedDynamics;
    using PreparedModel = variance_gamma::PreparedModel;
    using PreparedTransition = variance_gamma::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = variance_gamma::State;

    static constexpr bool kNativeLogSpot = true;

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
};

static_assert(equity::LogSpotDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::ExactTransitionDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::variance_gamma
