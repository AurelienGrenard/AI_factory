// CUDA analytics for G2++ fitted to Nelson-Siegel curves.
#pragma once

#include "curve/nelson_siegel/dataset.hpp"
#include "model/g2/dynamics.cuh"
#include "model/g2_plus_plus/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel {

// Correlated G2 process and initial curve defining one fitted G2++ model.
struct G2PlusPlusFittedParameters {
    model::g2::G2ProcessParameters process;
    curve::nelson_siegel::NelsonSiegelParameters initial_curve;
};

// Compose one G2++ row with its fitted initial curve.
__device__ __forceinline__ G2PlusPlusFittedParameters compose_model(
    const G2PlusPlusModelParameters& parameters,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve
);

// Return phi(t) in r(t) = x(t) + y(t) + phi(t).
__device__ __forceinline__ float short_rate_shift(
    const G2PlusPlusFittedParameters& parameters,
    float time
);

// Reconstruct the shifted short rate from both Gaussian states.
__device__ __forceinline__ float short_rate(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::G2State& state,
    float time
);

// The remaining analytics mirror the standalone G2 interface.

// Return the accumulated path log-discount from time zero.
__device__ __forceinline__ float log_discount_factor(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::joint::G2JointState& joint_state,
    float time
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::joint::G2JointState& joint_state,
    float time
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::G2State& state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::G2State& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::G2State& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate over [start_time,end_time].
__device__ __forceinline__ float forward_rate(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::G2State& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const G2PlusPlusFittedParameters& parameters,
    const model::g2::G2State& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
);

}  // namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel
