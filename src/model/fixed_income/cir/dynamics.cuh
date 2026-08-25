// Reusable CUDA interfaces for exact CIR state simulations.
#pragma once

#include "common/philox.cuh"
#include "common/simulation/concepts.cuh"
#include "model/fixed_income/cir/parameters.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::cir {

// Time-invariant coefficients of the noncentral chi-square law.
struct PreparedModel {
    float mean_reversion;
    float degrees_of_freedom;
    float scale_rate;
    float initial_state = 0.0f;
};

// Coefficients required to advance the prepared model by one delta_t.
struct PreparedTransition {
    float decay;
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

__device__ __forceinline__ float initial_state(
    const PreparedModel& model
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

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = cir::PreparedDynamics;
    using PreparedModel = cir::PreparedModel;
    using PreparedTransition = cir::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = float;

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
};

// Reserved for a future justified joint transition of r_t and its integral.
// The incomplete policy intentionally cannot satisfy a simulation concept.
namespace joint {
struct State {
    float state;
    float state_integral;
};
struct DynamicsPolicy;
}  // namespace joint

static_assert(simulation::DynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::ExactTransitionDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::model::cir
