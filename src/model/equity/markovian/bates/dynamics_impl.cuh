// Reusable Bates QE-M preparation and path simulation implementation.
#pragma once

#include "model/equity/markovian/bates/dynamics.cuh"

#include "common/philox.cuh"

// Bates composes the existing QE-M variance/spot transition without changing
// Heston's implementation or its random-number mapping.
#include "model/equity/markovian/heston/dynamics_impl.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::equity::bates {

// ======================== Common equity dynamics =========================

// Prepare the coefficients defining one transition of duration delta_t
// under the supplied model parameters.
__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float delta_t
) {
    const heston::ModelParameters heston_parameters = {
        parameters.spot,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        parameters.initial_variance,
        parameters.kappa,
        parameters.theta,
        parameters.gamma,
        parameters.rho,
    };
    const float poisson_mean = parameters.jump_intensity * delta_t;
    const float expected_relative_jump = expm1f(
        fmaf(
            0.5f * parameters.jump_log_volatility,
            parameters.jump_log_volatility,
            parameters.jump_log_mean
        )
    );

    return {
        heston::prepare_model(heston_parameters, delta_t),
        poisson_mean,
        expf(-poisson_mean),
        parameters.jump_log_mean,
        parameters.jump_log_volatility,
        parameters.jump_intensity * expected_relative_jump * delta_t,
    };
}

// Construct the time-zero state stored in the prepared QE parameters.
__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return heston::initial_state(prepared_model.heston);
}

// ==================== Model-specific implementation =======================

namespace {

// Add one already-sampled compound-Poisson increment to the log spot.
__device__ __forceinline__ void apply_jump_transition(
    const PreparedModel& prepared_model,
    float jump_compensator,
    std::uint32_t jump_count,
    float jump_normal,
    State& state
) {
    float jump_increment = -jump_compensator;
    if (jump_count != 0U) {
        const float count = static_cast<float>(jump_count);
        jump_increment = fmaf(count, prepared_model.jump_log_mean, jump_increment);
        jump_increment = fmaf(
            prepared_model.jump_log_volatility * sqrtf(count),
            jump_normal,
            jump_increment
        );
    }
    state.log_spot += jump_increment;
}

}  // namespace

// ======================== Common equity dynamics =========================

// Apply one variance and log-spot update with the QE-M martingale correction.
// Conditional on jump_count = n, the sum of n independent log jump sizes
// N(nu, delta^2) is represented exactly in law by
// n * nu + delta * sqrt(n) * jump_normal.  We therefore need neither the n
// individual sizes nor their event times to reproduce the process at grid
// dates.  This equivalence does not reconstruct the path inside the time step;
// continuously monitored or jump-time-dependent products need extra handling.
__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    std::uint32_t jump_count,
    float jump_normal,
    State& state
) {
    heston::one_step_transition(
        prepared_model.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        state
    );
    apply_jump_transition(
        prepared_model, prepared_model.jump_compensator, jump_count, jump_normal, state
    );
}

// ==================== Model-specific implementation =======================

namespace {

// Advance only the Heston component by one QE-M numerical step.
__device__ __forceinline__ void simulate_heston_one_step(
    const PreparedModel& prepared_model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    const float variance_normal = philox::next_normal(
        uniforms, normal_cache
    );
    const float stock_normal = philox::next_normal(
        uniforms, normal_cache
    );
    const float variance_uniform = uniforms.next();
    heston::one_step_transition(
        prepared_model.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        state
    );
}

// Draw and add the compound-Poisson increment over several equal QE steps.
// Heston is independent of the jump process and its variance does not depend
// on spot, so the jump sum may be applied after the Heston interval whenever
// the payoff observes only the interval boundary.
__device__ __forceinline__ void simulate_jump_interval(
    const PreparedModel& prepared_model,
    std::uint32_t step_count,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    const float count = static_cast<float>(step_count);
    const float poisson_mean = prepared_model.poisson_mean * count;
    const float zero_jump_probability = step_count == 1U
        ? prepared_model.zero_jump_probability
        : expf(-poisson_mean);
    const float jump_compensator = prepared_model.jump_compensator * count;

    const float poisson_uniform = uniforms.next();
    const std::uint32_t jump_count = philox::poisson_from_uniform(
        poisson_uniform,
        poisson_mean,
        zero_jump_probability
    );
    float jump_normal = 0.0f;
    if (jump_count != 0U) {
        jump_normal = philox::next_normal(uniforms, normal_cache);
    }
    apply_jump_transition(
        prepared_model, jump_compensator, jump_count, jump_normal, state
    );
}

// Simulate a boundary-only interval: daily Heston QE-M, then one exact jump sum.
__device__ __forceinline__ void simulate_interval(
    const PreparedModel& prepared_model,
    std::uint32_t step_count,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    if (step_count == 0U) return;
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        simulate_heston_one_step(prepared_model, uniforms, normal_cache, state);
    }
    simulate_jump_interval(
        prepared_model, step_count, uniforms, normal_cache, state
    );
}

// Draw one Bates transition from the continuous path-local uniform sequence.
// Conditional jump draws advance the same scalar stream without reservations.
__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& prepared_model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    State& state
) {
    simulate_heston_one_step(prepared_model, uniforms, normal_cache, state);
    simulate_jump_interval(prepared_model, 1U, uniforms, normal_cache, state);
}

}  // namespace

// ======================== Common equity dynamics =========================

// Generate all random variates for one path and return its terminal state.
__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    return bates::prepare_model(parameters, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return bates::initial_state(dynamics);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    bates::simulate_one_step(
        dynamics,
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
    bates::simulate_interval(
        dynamics,
        step_count,
        random.uniforms,
        random.normals,
        state
    );
}

__device__ __forceinline__ float DynamicsPolicy::spot(const State& state) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float DynamicsPolicy::log_spot(
    const State& state
) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::model::equity::bates
