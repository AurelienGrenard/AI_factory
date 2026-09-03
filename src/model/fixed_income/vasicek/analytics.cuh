// Public affine Vasicek analytics declarations used by fixed-income pricing bindings.
#pragma once

#include "common/fixed_income/mean_reverting_gaussian.cuh"
#include "common/fixed_income/cashflows.cuh"
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/vasicek/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::vasicek {

// Return the standalone short rate represented by the Vasicek state.
__device__ __forceinline__ float short_rate(
    const ModelParameters& parameters,
    float state,
    float time_years
);

// ===================== Common fixed-income analytics ======================

// Return the logarithm of the affine bond prefactor A(t,T).
__device__ __forceinline__ float log_A(
    const ModelParameters& parameters,
    float valuation_time_years,
    float maturity_years
);

// Return the affine bond prefactor A(t,T).
__device__ __forceinline__ float A(
    const ModelParameters& parameters,
    float valuation_time_years,
    float maturity_years
);

// Return the affine state loading B(t,T).
__device__ __forceinline__ float B(
    const ModelParameters& parameters,
    float valuation_time_years,
    float maturity_years
);

// Return log P(valuation_time_years,maturity_years) = log A - B*state.
__device__ __forceinline__ float log_zero_coupon_bond(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float maturity_years
);

// Return the accumulated path log-discount from time_years zero.
__device__ __forceinline__ float log_discount_factor(
    const ModelParameters& parameters,
    float state_integral,
    float time_years
);

// Return the accumulated path discount factor from time_years zero.
__device__ __forceinline__ float discount_factor(
    const ModelParameters& parameters,
    float state_integral,
    float time_years
);

// Return the model zero-coupon bond P(valuation_time_years, maturity_years).
__device__ __forceinline__ float zero_coupon_bond(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float maturity_years
);

// Return a call on P(option_expiry_years,bond_maturity_years), valued at valuation_time_years.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
);

// Return a put on P(option_expiry_years,bond_maturity_years), valued at valuation_time_years.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float option_expiry_years,
    float bond_maturity_years,
    float strike
);

// Return the simple forward rate observed at valuation_time_years over [start,end].
__device__ __forceinline__ float forward_rate(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    float end_time_years,
    float accrual_fraction
);

// Return the par swap rate observed at valuation_time_years.
template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    const ScheduleView& schedule
);

// Return the unit-notional value of the payer swap before optional exercise.
template<typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float start_time_years,
    float fixed_rate,
    const ScheduleView& schedule
);

// Solve the unique one-factor Jamshidian state boundary at exercise.
template<typename ScheduleView>
__device__ __forceinline__ float jamshidian_state_boundary(
    const ModelParameters& parameters,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Compatibility overload for one explicit day-count schedule.
__device__ __forceinline__ float jamshidian_state_boundary(
    const ModelParameters& parameters,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times_days,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
);

// Return P(exercise,payment; boundary), the Jamshidian bond strike.
__device__ __forceinline__ float jamshidian_bond_strike(
    const ModelParameters& parameters,
    float exercise_time,
    float payment_time,
    float state_boundary
);

template<SwaptionSide Side, typename ScheduleView>
__device__ __forceinline__ float european_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Price a unit-notional European payer swaption by Jamshidian decomposition.
template<typename ScheduleView>
__device__ __forceinline__ float european_payer_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Compatibility overload for one explicit day-count schedule.
__device__ __forceinline__ float european_payer_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times_days,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
);

// Price a unit-notional European receiver swaption by Jamshidian decomposition.
template<typename ScheduleView>
__device__ __forceinline__ float european_receiver_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Compatibility overload for one explicit day-count schedule.
__device__ __forceinline__ float european_receiver_swaption_price(
    const ModelParameters& parameters,
    float state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const std::uint32_t* __restrict__ payment_times_days,
    const float* __restrict__ accrual_fractions,
    float time_day_fraction,
    std::uint32_t payment_count
);

// Model-side analytics and regression projection used by Bermudan swaptions.
struct BermudanSwaptionAnalyticsPolicy {
    using PreparedModel = ModelParameters;

    struct PreparedRegressionState {
        float inverse_scale;
    };

    __device__ __forceinline__ static PreparedRegressionState
    prepare_regression_state(const ModelParameters& parameters) {
        return {
            ::ai_factory::workbench::fixed_income::
                mean_reverting_gaussian::
                    inverse_stationary_deviation_from_volatility(
                        parameters.process.volatility,
                        parameters.process.mean_reversion
                    ),
        };
    }

    __device__ __forceinline__ static float normalize_regression_state(
        const PreparedRegressionState& prepared,
        float state
    ) {
        return state * prepared.inverse_scale;
    }

    __device__ __forceinline__ static PreparedModel prepare_model(
        const ModelParameters& model
    ) {
        return model;
    }

    template<typename JointState>
    __device__ __forceinline__ static float factor_state(
        const JointState& state
    ) {
        return state.state;
    }

    __device__ __forceinline__ static float log_discount_factor(
        const PreparedModel& model,
        float state_integral,
        float time_years
    ) {
        return vasicek::log_discount_factor(model, state_integral, time_years);
    }

    template<typename ScheduleView>
    __device__ __forceinline__ static float payer_swap_value(
        const PreparedModel& model,
        float state,
        float valuation_time_years,
        float start_time_years,
        float fixed_rate,
        const ScheduleView& schedule
    ) {
        return vasicek::payer_swap_value(
            model,
            state,
            valuation_time_years,
            start_time_years,
            fixed_rate,
            schedule
        );
    }
};

}  // namespace ai_factory::workbench::model::fixed_income::vasicek
