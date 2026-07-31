// CUDA analytics for Hull-White one-factor fitted to Nelson-Siegel curves.
#pragma once

#include "curve/nelson_siegel/dataset.hpp"
#include "model/hull_white/dataset.hpp"
#include "model/ornstein_uhlenbeck/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::hull_white::nelson_siegel {

// OU parameters and fitted initial curve shared by one pricing row.
struct HullWhiteParameters {
    model::ornstein_uhlenbeck::OrnsteinUhlenbeckDynamicsParameters process;
    curve::nelson_siegel::NelsonSiegelParameters initial_curve;
};

// Compose one Hull-White row with its fitted initial curve.
__device__ __forceinline__ HullWhiteParameters prepare_model(
    const HullWhiteModelParameters& parameters,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve
);

// Return phi(t) in the shifted representation r(t) = x(t) + phi(t).
__device__ __forceinline__ float short_rate_shift(
    const HullWhiteParameters& parameters,
    float time
);

// Reconstruct the short rate from one simulated OU factor state.
__device__ __forceinline__ float short_rate(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float time
);

// Return the accumulated path log-discount from time zero.
__device__ __forceinline__ float log_discount(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float time
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float time
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate observed at valuation_time over [start,end].
__device__ __forceinline__ float forward_rate(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_period
);

// Return the par swap rate observed at valuation_time.
__device__ __forceinline__ float swap_rate(
    const HullWhiteParameters& parameters,
    const model::ornstein_uhlenbeck::OrnsteinUhlenbeckState& state,
    float valuation_time,
    float start_time,
    const float* __restrict__ payment_times,
    const float* __restrict__ accrual_periods,
    std::size_t payment_count
);

}  // namespace ai_factory::workbench::hull_white::nelson_siegel
