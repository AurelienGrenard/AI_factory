#pragma once

#include "model/equity/markovian/cev/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::cev {

// ======================== Common equity dynamics =========================

// Prepare the coefficients defining one transition of duration delta_t
// under the supplied model parameters.
__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters, float delta_t
) {
    return {
        parameters.spot,
        (parameters.risk_free_rate - parameters.dividend_yield) * delta_t,
        parameters.sigma * sqrtf(delta_t),
        0.5f * parameters.sigma * parameters.sigma * parameters.beta
            * delta_t,
        parameters.beta,
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return {prepared_model.initial_spot};
}

// Milstein preserves the local-volatility derivative term. Zero is absorbing,
// and a negative numerical proposal is projected to that attainable boundary.
__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model, float normal, State& state
) {
    const float spot = state.spot;
    if (!(spot > 0.0f)) {
        state.spot = 0.0f;
        return;
    }
    const float spot_beta = powf(spot, prepared_model.beta);
    // Reuse S^beta instead of evaluating a second power function:
    // S^(2 beta - 1) = (S^beta)^2 / S while S is strictly positive.
    const float spot_milstein_power = spot_beta * spot_beta / spot;
    const float proposal = spot
        + prepared_model.drift_dt * spot
        + prepared_model.diffusion_scale * spot_beta * normal
        + prepared_model.milstein_scale * spot_milstein_power * (normal * normal - 1.0f);
    state.spot = fmaxf(proposal, 0.0f);
}

// ======================== Common equity dynamics =========================

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    return cev::prepare_model(parameters, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return cev::initial_state(dynamics);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    cev::one_step_transition(
        dynamics,
        philox::next_normal(random.uniforms, random.normals),
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

__device__ __forceinline__ float DynamicsPolicy::spot(const State& state) {
    return state.spot;
}

__device__ __forceinline__ float DynamicsPolicy::log_spot(
    const State& state
) {
    return logf(state.spot);
}

}  // namespace ai_factory::workbench::model::equity::cev
