// Reusable CUDA interface for exact Black-Scholes transitions.
#pragma once

#include "equity/concepts.cuh"
#include "philox/distributions.cuh"
#include "model/equity/markovian/black_scholes/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::black_scholes {

struct State {
    float log_spot;
};

// Quantities computed once for one model and reused for every interval.
struct PreparedModel {
    float initial_log_spot;
    float drift_rate;
    float volatility;
};

// Exact Black-Scholes increment coefficients over one fixed interval.
struct PreparedTransition {
    float drift;
    float diffusion_standard_deviation;
};

struct PreparedDynamics {
    PreparedModel model;
    PreparedTransition transition;
};

// Physical Brownian shock over one standardized interval.
struct BrownianStep {
    float spot;
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
    float brownian_normal,
    State& state
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = black_scholes::PreparedDynamics;
    using PreparedModel = black_scholes::PreparedModel;
    using PreparedTransition = black_scholes::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = black_scholes::State;
    using BrownianStep = black_scholes::BrownianStep;

    static constexpr unsigned int kBrownianDimension = 1U;

    static constexpr bool kNativeLogSpot = true;
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
    template<typename AuxiliaryRandom>
    __device__ __forceinline__ static void one_step_transition(
        const PreparedDynamics& dynamics,
        BrownianStep brownian,
        AuxiliaryRandom&,
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
    // The model owns the point at which a shared short-rate integral enters
    // its numerical transition.
    template<typename AuxiliaryRandom>
    __device__ __forceinline__ static void one_step_transition_with_rate(
        const PreparedDynamics& dynamics, BrownianStep brownian,
        AuxiliaryRandom& auxiliary, State& state, float carry_shift
    ) {
        one_step_transition(dynamics, brownian, auxiliary, state);
        state.log_spot += carry_shift;
    }
    __device__ __forceinline__ static float spot(const State& state);
    __device__ __forceinline__ static float log_spot(const State& state);
};

static_assert(::ai_factory::workbench::equity::LogSpotDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::FixedStepDynamicsPolicy<DynamicsPolicy>);
static_assert(simulation::ExactTransitionDynamicsPolicy<DynamicsPolicy>);

}  // namespace ai_factory::workbench::model::equity::black_scholes
