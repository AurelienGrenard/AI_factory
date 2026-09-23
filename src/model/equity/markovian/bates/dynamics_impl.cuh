// Reusable Bates QE-M preparation and path simulation implementation.
#pragma once

#include "model/equity/markovian/bates/dynamics.cuh"

#include "common/compound_poisson.cuh"
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

__device__ __forceinline__ void apply_jump_interval(
    const PreparedModel& prepared_model,
    std::uint32_t step_count,
    std::uint32_t jump_count,
    float jump_standard_normal_sum,
    State& state
) {
    float jump_increment = -prepared_model.jump_compensator
        * static_cast<float>(step_count);
    if (jump_count != 0U) {
        const float count = static_cast<float>(jump_count);
        jump_increment = fmaf(count, prepared_model.jump_log_mean, jump_increment);
        jump_increment = fmaf(
            prepared_model.jump_log_volatility,
            jump_standard_normal_sum,
            jump_increment
        );
    }
    state.log_spot += jump_increment;
}

// ======================== Common equity dynamics =========================

// Apply one variance and log-spot update with the QE-M martingale correction.
// Conditional on jump_count = n, individual standard-normal marks are reduced
// to their running sum. This retains replayable event primitives without a
// mark array. Event times inside the interval are still not reconstructed;
// continuously monitored or jump-time-dependent products need extra handling.
__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    std::uint32_t jump_count,
    float jump_standard_normal_sum,
    State& state
) {
    heston::one_step_transition(
        prepared_model.heston,
        variance_normal,
        variance_uniform,
        stock_normal,
        state
    );
    apply_jump_interval(
        prepared_model, 1U, jump_count, jump_standard_normal_sum, state
    );
}

// ==================== Model-specific implementation =======================

namespace {

// Advance only the Heston component by one QE-M numerical step.
__device__ __forceinline__ void simulate_heston_one_step(
    const PreparedModel& prepared_model,
    philox::DomainRandomContext& random,
    State& state
) {
    const auto step = random.next_step();
    auto uniforms = random.source<compound_poisson::kContinuousSource>(step);
    philox::NormalPairCache normal_cache;
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
    philox::DomainRandomContext& random,
    std::uint32_t interval_start_step,
    State& state
) {
    auto count_uniforms = random.source<
        compound_poisson::kCountSource
    >(interval_start_step);
    const float count = static_cast<float>(step_count);
    const float poisson_mean = prepared_model.poisson_mean * count;
    const float zero_jump_probability = step_count == 1U
        ? prepared_model.zero_jump_probability
        : expf(-poisson_mean);
    const std::uint32_t jump_count = compound_poisson::draw_count(
        count_uniforms, poisson_mean, zero_jump_probability
    );
    auto mark_uniforms = random.source<
        compound_poisson::kMarkSource
    >(interval_start_step);
    philox::NormalPairCache mark_cache;
    float jump_standard_normal_sum = 0.0f;
    for (std::uint32_t event = 0U; event < jump_count; ++event) {
        jump_standard_normal_sum += philox::next_normal(
            mark_uniforms, mark_cache
        );
    }
    apply_jump_interval(
        prepared_model,
        step_count,
        jump_count,
        jump_standard_normal_sum,
        state
    );
}

// Simulate a boundary-only interval: daily Heston QE-M, then one exact jump sum.
__device__ __forceinline__ void simulate_interval(
    const PreparedModel& prepared_model,
    std::uint32_t step_count,
    philox::DomainRandomContext& random,
    State& state
) {
    if (step_count == 0U) return;
    const auto interval_start_step = random.step_index;
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        simulate_heston_one_step(prepared_model, random, state);
    }
    simulate_jump_interval(
        prepared_model, step_count, random, interval_start_step, state
    );
}

// Use the same step address for Heston and the independent jump sources.
__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& prepared_model,
    philox::DomainRandomContext& random,
    State& state
) {
    const auto step = random.step_index;
    simulate_heston_one_step(prepared_model, random, state);
    simulate_jump_interval(prepared_model, 1U, random, step, state);
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
        random,
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
        random,
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
