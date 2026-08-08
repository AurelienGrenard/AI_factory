// CUDA analytics for Hull-White one-factor fitted to Nelson-Siegel curves.
#pragma once

#include "curve/nelson_siegel/dataset.hpp"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::model::hull_white::nelson_siegel {

// OU process and initial curve defining one fitted Hull-White model.
struct HullWhiteFittedParameters {
    model::ornstein_uhlenbeck::OrnsteinUhlenbeckProcessParameters process;
    curve::nelson_siegel::NelsonSiegelParameters initial_curve;
};

// Compose one Hull-White row with its fitted initial curve.
__device__ __forceinline__ HullWhiteFittedParameters compose_model(
    const HullWhiteModelParameters& parameters,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve
);

// Return phi(t) in the shifted representation r(t) = x(t) + phi(t).
__device__ __forceinline__ float short_rate_shift(
    const HullWhiteFittedParameters& parameters,
    float time
);

// Reconstruct the short rate from one simulated OU state.
__device__ __forceinline__ float short_rate(
    const HullWhiteFittedParameters& parameters,
    float state,
    float time
);

// The remaining analytics mirror the standalone OU interface.

// Return the accumulated path log-discount from time zero.
__device__ __forceinline__ float log_discount_factor(
    const HullWhiteFittedParameters& parameters,
    const model::ornstein_uhlenbeck::joint::OrnsteinUhlenbeckJointState&
        joint_state,
    float time
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    const HullWhiteFittedParameters& parameters,
    const model::ornstein_uhlenbeck::joint::OrnsteinUhlenbeckJointState&
        joint_state,
    float time
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate observed at valuation_time over [start,end].
__device__ __forceinline__ float forward_rate(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
);

}  // namespace ai_factory::workbench::model::hull_white::nelson_siegel
