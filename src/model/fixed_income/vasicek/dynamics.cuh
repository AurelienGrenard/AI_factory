// Reusable CUDA interfaces for exact Vasicek simulations.
#pragma once

#include "common/philox.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::vasicek {

// E[integral r_s ds | r_t] = state_loading*r_t + mean_increment.
struct VasicekIntegralMoments {
    float state_loading;
    float mean_increment;
    float variance;
};

// Return B(delta), the loading of the current state in its future integral.
__device__ __forceinline__ float integral_state_loading(
    float mean_reversion,
    float delta
);

// Return the variance of the future integral of the Gaussian state.
__device__ __forceinline__ float integral_variance(
    const VasicekProcessParameters& parameters,
    float delta
);

// Compute both integral moments while sharing their exponential decay.
__device__ __forceinline__ VasicekIntegralMoments integral_moments(
    const VasicekProcessParameters& parameters,
    float delta
);

// Exact transition: r_next = decay*r + mean_increment + stddev*Z.
struct VasicekExactTransition {
    float decay;
    float mean_increment;
    float state_standard_deviation;
};

// Precompute the exact state transition reused at every equal time step.
__device__ __forceinline__ VasicekExactTransition prepare_model(
    const VasicekProcessParameters& parameters,
    float time_interval
);

// Advance the Vasicek state exactly over one time step.
__device__ __forceinline__ void one_step_transition(
    const VasicekExactTransition& model,
    float state_normal,
    float& state
);

// Apply the prepared exact transition and return its terminal state.
__device__ __forceinline__ float simulate_terminal_state(
    const VasicekExactTransition& model,
    float initial_state,
    float state_normal
);

// Store observation_count - 1 states; the terminal state is only returned.
// The date-major output requires (observation_count - 1) * path_count values.
__device__ __forceinline__ float simulate_on_regular_grid(
    const VasicekExactTransition& initial_stub_model,
    const VasicekExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states
);

// Joint Vasicek-state and time-integral simulation used for path discounting.
namespace joint {

// Exact affine Gaussian transition for r and its accumulated integral.
struct VasicekJointExactTransition {
    float decay;
    float state_mean_increment;
    float state_standard_deviation;
    float integral_state_loading;
    float integral_mean_increment;
    float integral_state_normal_loading;
    float integral_independent_standard_deviation;
};

// Store X_t and state_integral = integral_0^t X_s ds for one path.
struct VasicekJointState {
    float state;
    float state_integral;
};

// Precompute the exact joint transition reused at every equal time step.
__device__ __forceinline__ VasicekJointExactTransition prepare_model(
    const VasicekProcessParameters& parameters,
    float time_interval
);

// Advance the Vasicek state and its integral exactly over one time step.
__device__ __forceinline__ void one_step_transition(
    const VasicekJointExactTransition& model,
    float state_normal,
    float integral_normal,
    VasicekJointState& joint_state
);

// Apply the prepared exact transition and return its state and integral.
__device__ __forceinline__ VasicekJointState simulate_terminal_state(
    const VasicekJointExactTransition& model,
    float initial_state,
    float state_normal,
    float integral_normal
);

// Store observation_count - 1 joint states and return the terminal state.
// Each date-major output needs (observation_count - 1) * path_count values.
__device__ __forceinline__ VasicekJointState simulate_on_regular_grid(
    const VasicekJointExactTransition& initial_stub_model,
    const VasicekJointExactTransition& regular_model,
    float initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states,
    float* __restrict__ observed_integrated_states
);

}  // namespace joint
}  // namespace ai_factory::workbench::model::vasicek
