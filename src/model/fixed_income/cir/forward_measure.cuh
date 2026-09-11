// Public CIR terminal-forward dynamics and bond-numeraire analytics policies.
// Q dynamics and samples remain in dynamics.cuh; no rate integral is simulated here.
#pragma once

#include "model/fixed_income/cir/dynamics.cuh"

namespace ai_factory::workbench::model::fixed_income::cir::terminal_forward {

struct PreparedTransition {
    float scale;
    float state_loading;
};

__device__ __forceinline__ PreparedTransition prepare_transition(
    const ModelParameters& model, float interval_years, float remaining_to_numeraire_years
);

// The endpoint is scale * chi-square(df, state_loading * r / scale).
struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedModel = cir::PreparedModel;
    using PreparedTransition = terminal_forward::PreparedTransition;
    using State = float;
    using RandomContext = philox::NormalRandomContext;

    __device__ __forceinline__ static PreparedModel prepare_model(const Parameters&);
    __device__ __forceinline__ static State initial_state(const PreparedModel&);
    __device__ __forceinline__ static PreparedTransition prepare_transition(
        const Parameters&, float interval_years, float remaining_to_numeraire_years
    );
    __device__ __forceinline__ static void simulate_one_step(
        const PreparedModel&, const PreparedTransition&, RandomContext&, State&
    );
};

// Bond-numeraire analytics are independent of the transition sampler.
struct BondAnalyticsPolicy {
    __device__ __forceinline__ static float log_zero_coupon_bond(
        const ModelParameters&, float state, float time_years, float maturity_years
    );
    __device__ __forceinline__ static float log_A(const ModelParameters&, float time_years, float maturity_years);
    __device__ __forceinline__ static float B(const ModelParameters&, float time_years, float maturity_years);
};

}  // namespace ai_factory::workbench::model::fixed_income::cir::terminal_forward
