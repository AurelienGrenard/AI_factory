// Included device definitions of the exact CIR terminal-forward transition and numeraire policies.
#pragma once

#include "model/fixed_income/cir/forward_measure.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "model/fixed_income/cir/dynamics_impl.cuh"

namespace ai_factory::workbench::model::fixed_income::cir::terminal_forward {

__device__ __forceinline__ PreparedTransition prepare_transition(
    const ModelParameters& model, float interval_years, float remaining_to_numeraire_years
) {
    // Brigo--Mercurio, deterministic-shift extension, equations 17--18.
    // Negative exponentials avoid overflow; expm1 protects short intervals.
    const float kappa = model.process.mean_reversion;
    const float sigma = model.process.volatility;
    const float gamma = hypotf(kappa, 1.4142135623730951f * sigma);
    const float decay = expf(-gamma * interval_years);
    const float one_minus_decay = -expm1f(-gamma * interval_years);
    const float bond_loading = cir::B(model, 0.0f, remaining_to_numeraire_years);
    const float denominator = 2.0f * gamma * decay
        + (kappa + gamma + sigma * sigma * bond_loading) * one_minus_decay;
    const float root_loading = 2.0f * gamma * sqrtf(decay) / denominator;
    return {
        sigma * sigma * one_minus_decay / (2.0f * denominator),
        root_loading * root_loading,
    };
}

__device__ __forceinline__ DynamicsPolicy::PreparedModel
DynamicsPolicy::prepare_model(const Parameters& model) {
    return cir::DynamicsPolicy::prepare_model(model);
}

__device__ __forceinline__ DynamicsPolicy::State DynamicsPolicy::initial_state(
    const PreparedModel& model
) { return cir::initial_state(model); }

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(const Parameters& model, float interval_years, float remaining_to_numeraire_years) {
    return terminal_forward::prepare_transition(model, interval_years, remaining_to_numeraire_years);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedModel& model, const PreparedTransition& transition,
    RandomContext& random, State& state
) {
    state = philox::scaled_noncentral_chi_square(
        random.uniforms, random.normals, model.degrees_of_freedom,
        transition.state_loading * state / transition.scale, transition.scale
    );
}

__device__ __forceinline__ float BondAnalyticsPolicy::log_zero_coupon_bond(
    const ModelParameters& model, float state, float time_years, float maturity_years
) { return cir::log_zero_coupon_bond(model, state, time_years, maturity_years); }

__device__ __forceinline__ float BondAnalyticsPolicy::log_A(
    const ModelParameters& model, float time_years, float maturity_years
) { return cir::log_A(model, time_years, maturity_years); }

__device__ __forceinline__ float BondAnalyticsPolicy::B(
    const ModelParameters& model, float time_years, float maturity_years
) { return cir::B(model, time_years, maturity_years); }

}  // namespace ai_factory::workbench::model::fixed_income::cir::terminal_forward
