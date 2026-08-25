// CUDA analytics for Hull-White one-factor fitted to Nelson-Siegel curves.
#pragma once

#include "common/fixed_income/swaption_side.cuh"
#include "curve/nelson_siegel/dataset.hpp"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cuh"
#include "product/european_swaption/schedule.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::hull_white::nelson_siegel {

// ======================= Model-specific analytics =========================

// OU process and initial curve defining one fitted Hull-White model.
struct HullWhiteFittedParameters {
    model::ornstein_uhlenbeck::ProcessParameters process;
    curve::nelson_siegel::NelsonSiegelParameters initial_curve;
};

// Compose one Hull-White row with its fitted initial curve.
__device__ __forceinline__ HullWhiteFittedParameters compose_model(
    const ModelParameters& parameters,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve
);

// Model/curve composition consumed by generic closed-form pricing policies.
struct FittedModelComposition {
    using ModelParameters =
        ::ai_factory::workbench::model::hull_white::ModelParameters;
    using CurveParameters =
        ::ai_factory::workbench::curve::nelson_siegel::NelsonSiegelParameters;
    using FittedModel = HullWhiteFittedParameters;

    __device__ __forceinline__ static float initial_state() {
        return 0.0f;
    }

    __device__ __forceinline__ static FittedModel compose(
        const ModelParameters& model,
        const CurveParameters& initial_curve
    ) {
        return compose_model(model, initial_curve);
    }
};

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

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the affine bond prefactor A(t,T).
__device__ __forceinline__ float log_A(
    const HullWhiteFittedParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine bond prefactor A(t,T).
__device__ __forceinline__ float A(
    const HullWhiteFittedParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine OU-state loading B(t,T).
__device__ __forceinline__ float B(
    const HullWhiteFittedParameters& parameters,
    float valuation_time,
    float maturity
);

// Return log P(valuation_time,maturity) = log A - B*state.
__device__ __forceinline__ float log_zero_coupon_bond(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float maturity
);

// The remaining analytics mirror the standalone OU interface.

// Return the accumulated path log-discount from time zero.
__device__ __forceinline__ float log_discount_factor(
    const HullWhiteFittedParameters& parameters,
    float state_integral,
    float time
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    const HullWhiteFittedParameters& parameters,
    float state_integral,
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
template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    const ScheduleView& schedule
);

// Return the unit-notional value of the payer swap before optional exercise.
template<typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float start_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Solve the unique one-factor Jamshidian state boundary at exercise.
template<typename ScheduleView>
__device__ __forceinline__ float jamshidian_state_boundary(
    const HullWhiteFittedParameters& parameters,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Compatibility overload for one explicit day-count schedule.
__device__ __forceinline__ float jamshidian_state_boundary(
    const HullWhiteFittedParameters& parameters,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
);

// Return P(exercise,payment; boundary), the Jamshidian bond strike.
__device__ __forceinline__ float jamshidian_bond_strike(
    const HullWhiteFittedParameters& parameters,
    float exercise_time,
    float payment_time,
    float state_boundary
);

template<SwaptionSide Side, typename ScheduleView>
__device__ __forceinline__ float european_swaption_price(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Price a unit-notional European payer swaption by Jamshidian decomposition.
template<typename ScheduleView>
__device__ __forceinline__ float european_payer_swaption_price(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Compatibility overload for one explicit day-count schedule.
__device__ __forceinline__ float european_payer_swaption_price(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
);

// Price a unit-notional European receiver swaption by Jamshidian decomposition.
template<typename ScheduleView>
__device__ __forceinline__ float european_receiver_swaption_price(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Compatibility overload for one explicit day-count schedule.
__device__ __forceinline__ float european_receiver_swaption_price(
    const HullWhiteFittedParameters& parameters,
    float state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
);

}  // namespace ai_factory::workbench::model::hull_white::nelson_siegel
