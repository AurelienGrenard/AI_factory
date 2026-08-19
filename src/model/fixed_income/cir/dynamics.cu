// Exact CIR state simulation through the Poisson-Gamma representation.
#pragma once

#include "model/fixed_income/cir/dynamics.cuh"

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::cir {

// ======================== Model-specific dynamics =========================

// None: the adaptive exact law belongs directly in the common transition.

// ======================= Common state-only dynamics ========================

// Prepare the time-step constants shared by every state-dependent draw.
__device__ __forceinline__ CirExactTransition prepare_model(
    const CirProcessParameters& parameters,
    float time_interval
) {
    const float volatility_squared =
        parameters.volatility * parameters.volatility;
    const float one_minus_decay = -expm1f(
        -parameters.mean_reversion * time_interval
    );
    const float decay = 1.0f - one_minus_decay;
    return {
        decay,
        4.0f * parameters.mean_reversion * parameters.long_term_mean
            / volatility_squared,
        volatility_squared * one_minus_decay
            / (4.0f * parameters.mean_reversion),
    };
}

// Draw and apply one exact endpoint from the current path-local stream.
__device__ __forceinline__ void one_step_transition(
    const CirExactTransition& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    float& state
) {
    state = philox::scaled_noncentral_chi_square(
        uniforms,
        normal_cache,
        model.degrees_of_freedom,
        model.decay * state / model.scale,
        model.scale
    );
}

// Reuse one row key and one path-local scalar sequence for the exact draw.
__device__ __forceinline__ float simulate_terminal_state(
    const CirExactTransition& model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path
) {
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    one_step_transition(model, uniforms, normal_cache, initial_state);
    return initial_state;
}

// Write exact boundary states in a date-major grid and return the terminal one.
__device__ __forceinline__ float simulate_on_regular_grid(
    const CirExactTransition& initial_stub_model,
    const CirExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states
) {
    float state = initial_state;
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    // Reach the first observation through its possibly shorter stub.
    one_step_transition(
        initial_stub_model, uniforms, normal_cache, state
    );
    if (observation_count == 1U) return state;
    std::size_t output_index = path;
    observed_states[output_index] = state;

    // Store only pre-terminal states with one running date-major offset.
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        one_step_transition(
            regular_model, uniforms, normal_cache, state
        );
        output_index += path_count;
        observed_states[output_index] = state;
    }

    // The last exact transition is returned without a global-memory write.
    one_step_transition(regular_model, uniforms, normal_cache, state);
    return state;
}

namespace joint {

// ========================= Common joint dynamics ===========================

// Signatures are declared in dynamics.cuh. Definitions remain intentionally
// unavailable until the state-integral discretization has been validated.

}  // namespace joint

}  // namespace ai_factory::workbench::model::cir
