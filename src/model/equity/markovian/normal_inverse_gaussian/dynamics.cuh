// Reusable CUDA interface for exact Normal-Inverse-Gaussian increments.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox_domains.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/parameters.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

struct State {
    float log_spot;
};

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

struct TransitionInnovations {
    float inverse_gaussian_increment;
    float brownian_normal;
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
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    float inverse_gaussian_increment,
    float brownian_normal,
    State& state
);

__device__ __forceinline__ TransitionInnovations draw_transition_innovations(
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache
);

struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = normal_inverse_gaussian::PreparedDynamics;
    using PreparedModel = normal_inverse_gaussian::PreparedModel;
    using PreparedTransition = normal_inverse_gaussian::PreparedTransition;
    using RandomContext = philox::NormalRandomContext;
    using State = normal_inverse_gaussian::State;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kPartitionInvariantAdvance = true;
    static constexpr unsigned int kBrownianDimension = 1U;
    struct BrownianStep { float spot; };

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
    __device__ __forceinline__ static void one_step_transition(
        const PreparedDynamics& dynamics,
        BrownianStep brownian,
        philox::DomainUniformSequence& auxiliary,
        State& state
    );
    // The model owns the point at which a shared short-rate integral enters
    // its numerical transition.
    __device__ __forceinline__ static float nominal_rate_step(
        const Parameters& parameters, float delta_t
    ) {
        return parameters.risk_free_rate * delta_t;
    }
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
static_assert(simulation::CorrelatedStepDynamicsPolicy<
    DynamicsPolicy, philox::DomainUniformSequence>);

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
