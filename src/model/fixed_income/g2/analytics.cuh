// Public closed-form analytics declarations used by standalone G2 pricing bindings.
#pragma once

#include "common/fixed_income/mean_reverting_gaussian.cuh"
#include "model/fixed_income/g2/dynamics.cuh"
#include "model/fixed_income/g2/state.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::g2 {

// ======================= Model-specific analytics =========================

// Return the stochastic contribution x(t) + y(t).
__device__ __forceinline__ float stochastic_short_rate(const State& state);

// Return the short rate r(t) = x(t) + y(t).
__device__ __forceinline__ float short_rate(
    const ModelParameters& parameters,
    const State& state,
    float time
);

// ===================== Common fixed-income analytics ======================

// Two affine state loadings in P(t,T)=A(t,T)exp(-B_x*x-B_y*y).
struct TwoFactorAffineBondLoadings {
    float state_x;
    float state_y;
};

struct TwoFactorAffineBondCoefficients {
    float log_A;
    TwoFactorAffineBondLoadings B;
};

// Pure G2 primitives reused by standalone G2 and fitted G2++.
__device__ __forceinline__ TwoFactorAffineBondCoefficients
affine_bond_coefficients(
    const ProcessParameters& parameters,
    float time_to_maturity
);

__device__ __forceinline__ void state_covariances(
    const ProcessParameters& parameters,
    float time_to_expiry,
    float& variance_x,
    float& variance_y,
    float& covariance_xy
);

__device__ __forceinline__ float bond_option_total_volatility(
    const ProcessParameters& parameters,
    float time_to_expiry,
    float bond_tenor
);

// Return the logarithm of the affine bond prefactor A(t,T).
__device__ __forceinline__ float log_A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return the affine bond prefactor A(t,T).
__device__ __forceinline__ float A(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return both affine state loadings B(t,T).
__device__ __forceinline__ TwoFactorAffineBondLoadings B(
    const ModelParameters& parameters,
    float valuation_time,
    float maturity
);

// Return log P = log A - B_x*x - B_y*y.
__device__ __forceinline__ float log_zero_coupon_bond(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float maturity
);

// Return minus the accumulated short-rate integral from time zero.
__device__ __forceinline__ float log_discount_factor(
    const ModelParameters& parameters,
    float state_integral,
    float time
);

// Return the accumulated path discount factor from time zero.
__device__ __forceinline__ float discount_factor(
    const ModelParameters& parameters,
    float state_integral,
    float time
);

// Return the model zero-coupon bond P(valuation_time, maturity).
__device__ __forceinline__ float zero_coupon_bond(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float maturity
);

// Return a call on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_call_price(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return a put on P(option_expiry,bond_maturity), valued at valuation_time.
__device__ __forceinline__ float zero_coupon_bond_put_price(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float option_expiry,
    float bond_maturity,
    float strike
);

// Return the simple forward rate over [start_time,end_time].
__device__ __forceinline__ float forward_rate(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_fraction
);

// Return the par swap rate observed at valuation_time.
template<typename ScheduleView>
__device__ __forceinline__ float swap_rate(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    const ScheduleView& schedule
);

// Return the unit-notional value of the payer swap.
template<typename ScheduleView>
__device__ __forceinline__ float payer_swap_value(
    const ModelParameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    float fixed_rate,
    const ScheduleView& schedule
);

// Model-side analytics and regression projection used by Bermudan swaptions.
struct BermudanSwaptionAnalyticsPolicy {
    using PreparedModel = ModelParameters;

    struct PreparedRegressionState {
        float inverse_scale_x;
        float inverse_scale_y;
    };

    __device__ __forceinline__ static PreparedRegressionState
    prepare_regression_state(const ModelParameters& parameters) {
        return {
            ::ai_factory::workbench::fixed_income::
                mean_reverting_gaussian::
                    inverse_stationary_deviation_from_volatility(
                        parameters.process.volatility_x,
                        parameters.process.mean_reversion_x
                    ),
            ::ai_factory::workbench::fixed_income::
                mean_reverting_gaussian::
                    inverse_stationary_deviation_from_volatility(
                        parameters.process.volatility_y,
                        parameters.process.mean_reversion_y
                    ),
        };
    }

    __device__ __forceinline__ static float normalize_regression_state_x(
        const PreparedRegressionState& prepared,
        float state_x
    ) {
        return state_x * prepared.inverse_scale_x;
    }

    __device__ __forceinline__ static float normalize_regression_state_y(
        const PreparedRegressionState& prepared,
        float state_y
    ) {
        return state_y * prepared.inverse_scale_y;
    }

    __device__ __forceinline__ static PreparedModel prepare_model(
        const ModelParameters& model
    ) {
        return model;
    }

    template<typename JointState>
    __device__ __forceinline__ static State factor_state(
        const JointState& state
    ) {
        return state.state;
    }

    __device__ __forceinline__ static float log_discount_factor(
        const PreparedModel& model,
        float state_integral,
        float time
    ) {
        return g2::log_discount_factor(model, state_integral, time);
    }

    template<typename ScheduleView>
    __device__ __forceinline__ static float payer_swap_value(
        const PreparedModel& model,
        const State& state,
        float valuation_time,
        float start_time,
        float fixed_rate,
        const ScheduleView& schedule
    ) {
        return g2::payer_swap_value(
            model,
            state,
            valuation_time,
            start_time,
            fixed_rate,
            schedule
        );
    }
};

}  // namespace ai_factory::workbench::model::fixed_income::g2
