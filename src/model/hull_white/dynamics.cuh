// Reusable CUDA interface for exact Hull-White one-factor dynamics.
#pragma once

#include "common/philox.cuh"
#include "curve/nelson_siegel/dataset.hpp"
#include "model/hull_white/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::hull_white {

// Gaussian factor and its accumulated integral private to one path.
struct HullWhiteState {
    float factor;
    float integrated_factor;
};

// Exact joint-transition coefficients reused at every equal time step.
struct HullWhiteStepParameters {
    float dt;
    float decay;
    float factor_standard_deviation;
    float integral_factor_loading;
    float integral_factor_normal_loading;
    float integral_independent_standard_deviation;
};

// Precompute the exact factor and factor-integral transition coefficients.
__device__ __forceinline__ HullWhiteStepParameters prepare_hull_white_step(
    const HullWhiteOneFactorParameters& parameters,
    float dt
);

// Construct the deterministic time-zero factor and integral state.
__device__ __forceinline__ HullWhiteState initial_hull_white_state();

// Return phi(t) in the shifted representation r(t) = x(t) + phi(t).
__device__ __forceinline__ float hull_white_short_rate_shift(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    float time
);

// Return the path short rate from the current factor and initial curve.
__device__ __forceinline__ float hull_white_short_rate(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float time
);

// Return the path log-discount from its factor integral and initial curve.
__device__ __forceinline__ float hull_white_log_discount(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float time
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float hull_white_discount_factor(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float time
);

// Return theta(t) in dr = (theta(t) - a r) dt + sigma dW.
__device__ __forceinline__ float hull_white_drift_level(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    float time
);

// Advance the factor and its integral exactly over one time step.
__device__ __forceinline__ void one_step_hull_white_transition(
    const HullWhiteStepParameters& step,
    float factor_normal,
    float integral_normal,
    HullWhiteState& state
);

// Simulate one complete path and return its terminal factor and integral.
__device__ __forceinline__ HullWhiteState simulate_terminal_hull_white_state(
    const HullWhiteStepParameters& step,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

// Store pre-maturity factor and integral states on a regular exercise grid.
__device__ __forceinline__ HullWhiteState
simulate_factor_discount_on_regular_grid(
    const HullWhiteStepParameters& initial_stub_step,
    const HullWhiteStepParameters& regular_step,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_factors,
    float* __restrict__ observed_integrated_factors
);

// Return the model zero-coupon bond P(t,T) from the current factor.
__device__ __forceinline__ float hull_white_zero_coupon_bond(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float valuation_time,
    float maturity
);

}  // namespace ai_factory::workbench::hull_white
