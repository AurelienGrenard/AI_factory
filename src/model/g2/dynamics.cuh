// Reusable CUDA interfaces for exact correlated G2 simulations.
#pragma once

#include "common/philox.cuh"
#include "model/g2/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::g2 {

// Conditional moments of integral_t^{t+d}(X_s + Y_s) ds.
struct G2IntegralMoments {
    float state_x_loading;
    float state_y_loading;
    float variance;
};

// Compute both state loadings and the variance of the future rate integral.
__device__ __forceinline__ G2IntegralMoments integral_moments(
    const G2ProcessParameters& parameters,
    float delta
);

// Exact correlated transition of the two Gaussian factor states.
struct G2ExactTransition {
    float decay_x;
    float state_x_standard_deviation;
    float decay_y;
    float state_y_x_normal_loading;
    float state_y_independent_standard_deviation;
};

// Precompute one exact two-factor transition over a fixed interval.
__device__ __forceinline__ G2ExactTransition prepare_model(
    const G2ProcessParameters& parameters,
    float time_interval
);

// Advance both correlated Gaussian states exactly over one interval.
__device__ __forceinline__ void one_step_transition(
    const G2ExactTransition& model,
    float state_x_normal,
    float state_y_normal,
    G2State& state
);

// Apply one prepared exact transition and return both terminal states.
__device__ __forceinline__ G2State simulate_terminal_state(
    const G2ExactTransition& model,
    G2State initial_state,
    float state_x_normal,
    float state_y_normal
);

// Store observation_count - 1 states and return the terminal state.
__device__ __forceinline__ G2State simulate_on_regular_grid(
    const G2ExactTransition& initial_stub_model,
    const G2ExactTransition& regular_model,
    G2State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y
);

// Joint two-factor state and accumulated short-rate integral simulation.
namespace joint {

// Cholesky coefficients for the exact joint transition of X, Y, and integral.
struct G2JointExactTransition {
    float decay_x;
    float state_x_standard_deviation;
    float decay_y;
    float state_y_x_normal_loading;
    float state_y_independent_standard_deviation;
    float integral_state_x_loading;
    float integral_state_y_loading;
    float integral_x_normal_loading;
    float integral_y_normal_loading;
    float integral_independent_standard_deviation;
};

// Store both factor states and integral_0^t (X_s + Y_s) ds.
struct G2JointState {
    G2State state;
    float state_integral;
};

// Precompute the exact joint transition over one fixed interval.
__device__ __forceinline__ G2JointExactTransition prepare_model(
    const G2ProcessParameters& parameters,
    float time_interval
);

// Advance the two states and their accumulated integral exactly.
__device__ __forceinline__ void one_step_transition(
    const G2JointExactTransition& model,
    float state_x_normal,
    float state_y_normal,
    float integral_normal,
    G2JointState& joint_state
);

// Apply one prepared joint transition and return its terminal state.
__device__ __forceinline__ G2JointState simulate_terminal_state(
    const G2JointExactTransition& model,
    G2State initial_state,
    float state_x_normal,
    float state_y_normal,
    float integral_normal
);

// Store pre-terminal joint states and return the terminal state.
__device__ __forceinline__ G2JointState simulate_on_regular_grid(
    const G2JointExactTransition& initial_stub_model,
    const G2JointExactTransition& regular_model,
    G2State initial_state,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_states_x,
    float* __restrict__ observed_states_y,
    float* __restrict__ observed_integrated_states
);

}  // namespace joint
}  // namespace ai_factory::workbench::model::g2
