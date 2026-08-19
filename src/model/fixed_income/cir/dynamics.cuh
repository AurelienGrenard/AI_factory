// Reusable CUDA interfaces for exact CIR state simulations.
#pragma once

#include "common/philox.cuh"
#include "model/fixed_income/cir/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::cir {

// ======================== Model-specific dynamics =========================

// None: the adaptive exact law belongs directly in the common transition.

// ======================= Common state-only dynamics ========================

// Exact transition: r_next = scale * chi_square(df, noncentrality(r)).
struct CirExactTransition {
    float decay;
    float degrees_of_freedom;
    float scale;
};

// Precompute the exact state transition reused at every equal time step.
__device__ __forceinline__ CirExactTransition prepare_model(
    const CirProcessParameters& parameters,
    float time_interval
);

// Draw and apply one exact state transition on the current path stream.
__device__ __forceinline__ void one_step_transition(
    const CirExactTransition& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    float& state
);

// Apply the prepared exact transition and return its terminal state.
__device__ __forceinline__ float simulate_terminal_state(
    const CirExactTransition& model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path
);

// Store observation_count - 1 states; the terminal state is only returned.
// The date-major output requires (observation_count - 1) * path_count values.
__device__ __forceinline__ float simulate_on_regular_grid(
    const CirExactTransition& initial_stub_model,
    const CirExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states
);

namespace joint {

// ========================= Common joint dynamics ===========================

// The numerical joint transition is deliberately left unspecified until its
// state-integral approximation and validation contract have been selected.
struct CirJointTransition;

struct CirJointState {
    float state;
    float state_integral;
};

__device__ __forceinline__ CirJointTransition prepare_model(
    const CirProcessParameters& parameters,
    float time_interval
);

__device__ __forceinline__ void one_step_transition(
    const CirJointTransition& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    CirJointState& joint_state
);

__device__ __forceinline__ CirJointState simulate_terminal_state(
    const CirJointTransition& model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ CirJointState simulate_on_regular_grid(
    const CirJointTransition& initial_stub_model,
    const CirJointTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
);

}  // namespace joint

}  // namespace ai_factory::workbench::model::cir
