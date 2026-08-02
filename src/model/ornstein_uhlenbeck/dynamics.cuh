// Reusable CUDA interfaces for exact Ornstein-Uhlenbeck simulations.
#pragma once

#include "common/philox.cuh"
#include "model/ornstein_uhlenbeck/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// For I = integral_t^{t+d} X_s ds: E[I|X_t] = state_loading * X_t,
// while variance is Var[I|X_t].
struct OrnsteinUhlenbeckIntegralMoments {
    float state_loading;
    float variance;
};

// Return B(delta), the loading of the current state in its future integral.
__device__ __forceinline__ float integral_state_loading(
    float mean_reversion,
    float delta
);

// Return the variance of the future integral of the Gaussian state.
__device__ __forceinline__ float integral_variance(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float delta
);

// Compute both integral moments while sharing their exponential decay.
__device__ __forceinline__ OrnsteinUhlenbeckIntegralMoments integral_moments(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float delta
);

// Exact transition: X_next = decay * X + state_standard_deviation * Z.
struct OrnsteinUhlenbeckExactTransition {
    float decay;
    float state_standard_deviation;
};

// Precompute the exact state transition reused at every equal time step.
__device__ __forceinline__ OrnsteinUhlenbeckExactTransition prepare_model(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float time_interval
);

// Advance the OU state exactly over one time step.
__device__ __forceinline__ void one_step_transition(
    const OrnsteinUhlenbeckExactTransition& model,
    float state_normal,
    float& state
);

// Apply the prepared exact transition and return its terminal state.
__device__ __forceinline__ float simulate_terminal_state(
    const OrnsteinUhlenbeckExactTransition& model,
    float initial_state,
    float state_normal
);

// Store observation_count - 1 states; the terminal state is only returned.
// The date-major output requires (observation_count - 1) * path_count values.
__device__ __forceinline__ float simulate_on_regular_grid(
    const OrnsteinUhlenbeckExactTransition& initial_stub_model,
    const OrnsteinUhlenbeckExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states
);

// Joint OU-state and time-integral simulation used for path discounting.
namespace joint {

// X_next = decay*X + state_standard_deviation*Z1; I_next = I + integral_state_loading*X
// + integral_state_normal_loading*Z1 + integral_independent_standard_deviation*Z2.
struct OrnsteinUhlenbeckJointExactTransition {
    float decay;
    float state_standard_deviation;
    float integral_state_loading;
    float integral_state_normal_loading;
    float integral_independent_standard_deviation;
};

// Store X_t and state_integral = integral_0^t X_s ds for one path.
struct OrnsteinUhlenbeckJointState {
    float state;
    float state_integral;
};

// Precompute the exact joint transition reused at every equal time step.
__device__ __forceinline__ OrnsteinUhlenbeckJointExactTransition prepare_model(
    const OrnsteinUhlenbeckProcessParameters& parameters,
    float time_interval
);

// Advance the OU state and its integral exactly over one time step.
__device__ __forceinline__ void one_step_transition(
    const OrnsteinUhlenbeckJointExactTransition& model,
    float state_normal,
    float integral_normal,
    OrnsteinUhlenbeckJointState& joint_state
);

// Apply the prepared exact transition and return its state and integral.
__device__ __forceinline__ OrnsteinUhlenbeckJointState simulate_terminal_state(
    const OrnsteinUhlenbeckJointExactTransition& model,
    float initial_state,
    float state_normal,
    float integral_normal
);

// Store observation_count - 1 joint states and return the terminal state.
// Each date-major output needs (observation_count - 1) * path_count values.
__device__ __forceinline__ OrnsteinUhlenbeckJointState simulate_on_regular_grid(
    const OrnsteinUhlenbeckJointExactTransition& initial_stub_model,
    const OrnsteinUhlenbeckJointExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
);

}  // namespace joint
}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
