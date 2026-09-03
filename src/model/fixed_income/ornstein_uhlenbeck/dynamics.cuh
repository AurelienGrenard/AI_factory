// Reusable CUDA interfaces for exact Ornstein-Uhlenbeck simulations.
#pragma once

#include "common/philox.cuh"
#include "common/simulation/concepts.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {

using State = float;

struct IntegralMoments {
    float state_loading;
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

// Time-invariant coefficients of one OU process.
struct PreparedModel {
    float mean_reversion;
    float volatility_squared;
    float initial_state = 0.0f;
};

// Coefficients required to advance the prepared model by one delta_t.
struct PreparedTransition {
    float state_decay;
    float state_standard_deviation;
};

struct PreparedDynamics {
    PreparedModel model;
    PreparedTransition transition;
};

__device__ __forceinline__ PreparedModel prepare_model(
    const ProcessParameters& parameters
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
    float state_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = ornstein_uhlenbeck::PreparedDynamics;
    using PreparedModel = ornstein_uhlenbeck::PreparedModel;
    using PreparedTransition = ornstein_uhlenbeck::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = ornstein_uhlenbeck::State;

    static constexpr bool kPartitionInvariantAdvance = true;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters,
        float delta_t
    );
    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters
    );
    __device__ __forceinline__ static PreparedTransition prepare_transition(
        const PreparedModel& prepared_model,
        float delta_t
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedModel& prepared_model
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
        const PreparedModel& prepared_model,
        const PreparedTransition& prepared_transition,
        RandomContext& random,
        State& state
    );
};

namespace joint {

struct State {
    float state;
    float state_integral;
};

struct PreparedTransition {
    float state_decay;
    float state_standard_deviation;
    float integral_state_loading;
    float integral_state_normal_loading;
    float integral_independent_standard_deviation;
};

struct PreparedDynamics {
    ornstein_uhlenbeck::PreparedModel model;
    PreparedTransition transition;
};

__device__ __forceinline__ PreparedTransition prepare_transition(
    const ornstein_uhlenbeck::PreparedModel& prepared_model,
    float delta_t
);

__device__ __forceinline__ State initial_state(
    const ornstein_uhlenbeck::PreparedModel& prepared_model
);

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& prepared_transition,
    float state_normal,
    float integral_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = joint::PreparedDynamics;
    using PreparedModel = ornstein_uhlenbeck::PreparedModel;
    using PreparedTransition = joint::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = joint::State;

    static constexpr bool kPartitionInvariantAdvance = true;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters,
        float delta_t
    );
    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters
    );
    __device__ __forceinline__ static PreparedTransition prepare_transition(
        const PreparedModel& prepared_model,
        float delta_t
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedModel& prepared_model
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
        const PreparedModel& prepared_model,
        const PreparedTransition& prepared_transition,
        RandomContext& random,
        State& state
    );
};

}  // namespace joint

static_assert(simulation::DynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::ExactTransitionDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::DynamicsPolicy<joint::DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<joint::DynamicsPolicy>);
static_assert(simulation::ExactTransitionDynamicsPolicy<joint::DynamicsPolicy>);

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
