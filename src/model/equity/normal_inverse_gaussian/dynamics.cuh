// Reusable CUDA interface for exact Normal-Inverse-Gaussian increments.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "model/equity/normal_inverse_gaussian/parameters.hpp"

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

struct PreparedDynamics {
    PreparedModel model;
    PreparedTransition transition;
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
    float inverse_gaussian_increment,
    float brownian_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = normal_inverse_gaussian::PreparedDynamics;
    using PreparedModel = normal_inverse_gaussian::PreparedModel;
    using PreparedTransition = normal_inverse_gaussian::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = normal_inverse_gaussian::State;

    static constexpr bool kNativeLogSpot = true;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters, float delta_t
    );
    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters
    );
    __device__ __forceinline__ static PreparedTransition prepare_transition(
        const PreparedModel& model, float delta_t
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedModel& model
    );
    __device__ __forceinline__ static void simulate_one_step(
        const PreparedDynamics& dynamics, RandomContext& random, State& state
    );
    __device__ __forceinline__ static void advance(
        const PreparedDynamics& dynamics, std::uint32_t step_count,
        RandomContext& random, State& state
    );
    __device__ __forceinline__ static void simulate_one_step(
        const PreparedModel& model, const PreparedTransition& transition,
        RandomContext& random, State& state
    );
    __device__ __forceinline__ static float spot(const State& state);
    __device__ __forceinline__ static float log_spot(const State& state);
    __device__ __forceinline__ static float risk_free_rate(
        const Parameters& parameters
    );
};

static_assert(equity::EquityDynamicsPolicy<DynamicsPolicy>);
static_assert(equity::ExactTransitionDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::normal_inverse_gaussian
