// Reusable CUDA interfaces for exact correlated G2 simulations.
#pragma once

#include "common/philox.cuh"
#include "common/simulation/concepts.cuh"
#include "model/fixed_income/g2/parameters.hpp"
#include "model/fixed_income/g2/state.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::g2 {

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
    State initial_state = {0.0f, 0.0f};
};

struct PreparedTransition {
    float state_x_decay;
    float state_x_standard_deviation;
    float state_y_decay;
    float state_y_x_normal_loading;
    float state_y_independent_standard_deviation;
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
    float state_x_normal,
    float state_y_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = g2::PreparedDynamics;
    using PreparedModel = g2::PreparedModel;
    using PreparedTransition = g2::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = g2::State;

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
    g2::State state;
    float state_integral;
};

struct PreparedTransition {
    float state_x_decay;
    float state_x_standard_deviation;
    float state_y_decay;
    float state_y_x_normal_loading;
    float state_y_independent_standard_deviation;
    float integral_state_x_loading;
    float integral_state_y_loading;
    float integral_x_normal_loading;
    float integral_y_normal_loading;
    float integral_independent_standard_deviation;
};

struct PreparedDynamics {
    g2::PreparedModel model;
    PreparedTransition transition;
};

__device__ __forceinline__ PreparedTransition prepare_transition(
    const g2::PreparedModel& prepared_model,
    float delta_t
);

__device__ __forceinline__ State initial_state(
    const g2::PreparedModel& prepared_model
);

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& prepared_transition,
    float state_x_normal,
    float state_y_normal,
    float integral_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = joint::PreparedDynamics;
    using PreparedModel = g2::PreparedModel;
    using PreparedTransition = joint::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = joint::State;

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

}  // namespace ai_factory::workbench::model::fixed_income::g2
