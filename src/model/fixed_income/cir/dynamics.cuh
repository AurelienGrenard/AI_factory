// Reusable CUDA interfaces for exact CIR state simulations.
#pragma once

#include "common/philox.cuh"
#include "common/simulation/concepts.cuh"
#include "model/fixed_income/cir/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::cir {

using State = float;

// Time-invariant coefficients of the noncentral chi-square law.
struct PreparedModel {
    float mean_reversion;
    float degrees_of_freedom;
    float scale_rate;
    float initial_state = 0.0f;
};

// Coefficients required to advance the prepared model by one delta_t.
struct PreparedTransition {
    float state_decay;
    float scale;
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
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = cir::PreparedDynamics;
    using PreparedModel = cir::PreparedModel;
    using PreparedTransition = cir::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = cir::State;

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
    float state;
    float state_integral;
};

// Exact CIR state transition plus trapezoidal integral accumulation.
struct PreparedTransition {
    cir::PreparedTransition state_transition;
    float integral_trapezoid_scale;
};

struct PreparedDynamics {
    cir::PreparedModel model;
    PreparedTransition transition;
};

__device__ __forceinline__ PreparedTransition prepare_transition(
    const cir::PreparedModel& prepared_model,
    float delta_t
);

__device__ __forceinline__ State initial_state(
    const cir::PreparedModel& prepared_model
);

__device__ __forceinline__ void one_step_transition(
    const cir::PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = joint::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = joint::State;

    static constexpr bool kPartitionInvariantAdvance = true;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters, float delta_t
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
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
};

}  // namespace joint

static_assert(simulation::DynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::ExactTransitionDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::DynamicsPolicy<joint::DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<joint::DynamicsPolicy>);
static_assert(!simulation::ExactTransitionDynamicsPolicy<
    joint::DynamicsPolicy
>);

}  // namespace ai_factory::workbench::model::fixed_income::cir
