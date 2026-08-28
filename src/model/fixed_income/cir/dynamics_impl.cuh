// Included device definitions for exact CIR simulation through the Poisson-Gamma representation.
#pragma once

#include "model/fixed_income/cir/dynamics.cuh"

namespace ai_factory::workbench::model::fixed_income::cir {

__device__ __forceinline__ PreparedModel prepare_model(
    const ProcessParameters& parameters
) {
    const float volatility_squared =
        parameters.volatility * parameters.volatility;
    return {
        parameters.mean_reversion,
        4.0f * parameters.mean_reversion * parameters.long_term_mean
            / volatility_squared,
        volatility_squared / (4.0f * parameters.mean_reversion),
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
) {
    const float one_minus_decay = -expm1f(-prepared_model.mean_reversion * delta_t);
    return {1.0f - one_minus_decay, prepared_model.scale_rate * one_minus_decay};
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return prepared_model.initial_state;
}

__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    state = philox::scaled_noncentral_chi_square(
        uniforms,
        normal_cache,
        prepared_model.degrees_of_freedom,
        prepared_transition.state_decay * state / prepared_transition.scale,
        prepared_transition.scale
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    one_step_transition(
        prepared_model,
        prepared_transition,
        uniforms,
        normal_cache,
        state
    );
}

}  // namespace

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters, float delta_t
) {
    const PreparedModel prepared_model =
        DynamicsPolicy::prepare_model(parameters);
    return {
        prepared_model,
        DynamicsPolicy::prepare_transition(prepared_model, delta_t),
    };
}

__device__ __forceinline__ DynamicsPolicy::PreparedModel
DynamicsPolicy::prepare_model(const Parameters& parameters) {
    PreparedModel prepared_model = cir::prepare_model(parameters.process);
    prepared_model.initial_state = parameters.initial_state;
    return prepared_model;
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& prepared_model, float delta_t
) {
    return cir::prepare_transition(prepared_model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return cir::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& prepared_model) {
    return cir::initial_state(prepared_model);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    DynamicsPolicy::simulate_one_step(
        dynamics.model, dynamics.transition, random, state
    );
}

__device__ __forceinline__ void DynamicsPolicy::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        DynamicsPolicy::simulate_one_step(dynamics, random, state);
    }
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    RandomContext& random,
    State& state
) {
    cir::simulate_one_step(
        prepared_model, prepared_transition, random.uniforms, random.normals, state
    );
}

namespace joint {

__device__ __forceinline__ PreparedTransition prepare_transition(
    const cir::PreparedModel& prepared_model,
    float delta_t
) {
    return {
        cir::prepare_transition(prepared_model, delta_t),
        0.5f * delta_t,
    };
}

__device__ __forceinline__ State initial_state(
    const cir::PreparedModel& prepared_model
) {
    return {cir::initial_state(prepared_model), 0.0f};
}

__device__ __forceinline__ void one_step_transition(
    const cir::PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    const float previous_state = state.state;
    cir::one_step_transition(
        prepared_model,
        prepared_transition.state_transition,
        uniforms,
        normal_cache,
        state.state
    );
    state.state_integral = fmaf(
        prepared_transition.integral_trapezoid_scale,
        previous_state + state.state,
        state.state_integral
    );
}

namespace {

__device__ __forceinline__ void simulate_one_step(
    const cir::PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    one_step_transition(
        prepared_model,
        prepared_transition,
        uniforms,
        normal_cache,
        state
    );
}

}  // namespace

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    const cir::PreparedModel prepared_model =
        cir::DynamicsPolicy::prepare_model(parameters);
    return {prepared_model, joint::prepare_transition(prepared_model, delta_t)};
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return joint::initial_state(dynamics.model);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    joint::simulate_one_step(
        dynamics.model,
        dynamics.transition,
        random.uniforms,
        random.normals,
        state
    );
}

__device__ __forceinline__ void DynamicsPolicy::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        DynamicsPolicy::simulate_one_step(dynamics, random, state);
    }
}

}  // namespace joint

}  // namespace ai_factory::workbench::model::fixed_income::cir
