// Static parameter adapters for reusing one process under another model row.
#pragma once

#include "common/simulation/concepts.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::simulation {

template<
    typename OuterParameters,
    ExactTransitionDynamicsPolicy BaseDynamics,
    typename ParameterAdapter
>
struct AdaptedExactTransitionDynamicsPolicy {
    using Parameters = OuterParameters;
    using PreparedDynamics = typename BaseDynamics::PreparedDynamics;
    using PreparedModel = typename BaseDynamics::PreparedModel;
    using PreparedTransition = typename BaseDynamics::PreparedTransition;
    using RandomContext = typename BaseDynamics::RandomContext;
    using State = typename BaseDynamics::State;

    static constexpr bool kPartitionInvariantAdvance =
        BaseDynamics::kPartitionInvariantAdvance;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Parameters& parameters,
        float delta_t
    ) {
        return BaseDynamics::prepare_dynamics(
            ParameterAdapter::adapt(parameters), delta_t
        );
    }

    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters
    ) {
        return BaseDynamics::prepare_model(
            ParameterAdapter::adapt(parameters)
        );
    }

    __device__ __forceinline__ static PreparedTransition prepare_transition(
        const PreparedModel& model,
        float delta_t
    ) {
        return BaseDynamics::prepare_transition(model, delta_t);
    }

    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    ) {
        return BaseDynamics::initial_state(dynamics);
    }

    __device__ __forceinline__ static State initial_state(
        const PreparedModel& model
    ) {
        return BaseDynamics::initial_state(model);
    }

    __device__ __forceinline__ static void simulate_one_step(
        const PreparedDynamics& dynamics,
        RandomContext& random,
        State& state
    ) {
        BaseDynamics::simulate_one_step(dynamics, random, state);
    }

    __device__ __forceinline__ static void advance(
        const PreparedDynamics& dynamics,
        std::uint32_t step_count,
        RandomContext& random,
        State& state
    ) {
        BaseDynamics::advance(dynamics, step_count, random, state);
    }

    __device__ __forceinline__ static void simulate_one_step(
        const PreparedModel& model,
        const PreparedTransition& transition,
        RandomContext& random,
        State& state
    ) {
        BaseDynamics::simulate_one_step(model, transition, random, state);
    }
};

}  // namespace ai_factory::workbench::simulation
