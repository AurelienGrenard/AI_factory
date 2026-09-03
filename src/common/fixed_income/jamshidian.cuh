// One-factor Jamshidian boundary and European-swaption decomposition.
#pragma once

#include "common/fixed_income/one_factor_affine.cuh"
#include "common/fixed_income/swaption_side.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace ai_factory::workbench::fixed_income {

inline constexpr std::uint32_t kMaximumJamshidianNewtonIterations = 48U;
inline constexpr float kJamshidianResidualTolerance = 2.0e-7f;

struct JamshidianBoundaryEvaluation {
    float residual;
    float derivative;
};

// Return c_i = K*delta_i plus the final unit redemption.
__device__ __forceinline__ float jamshidian_cashflow_coefficient(
    float fixed_rate,
    float accrual_fraction,
    std::uint32_t payment,
    std::uint32_t payment_count
) {
    return fmaf(
        fixed_rate,
        accrual_fraction,
        payment + 1U == payment_count ? 1.0f : 0.0f
    );
}

template<typename Provider, typename Parameters, typename ScheduleView>
__device__ __forceinline__ JamshidianBoundaryEvaluation
evaluate_jamshidian_boundary(
    const Provider& provider,
    const Parameters& parameters,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule,
    float state
) {
    float coupon_bond = 0.0f;
    float derivative = 0.0f;
    const std::uint32_t payment_count = schedule.payment_count();
    for (std::uint32_t payment = 0U;
         payment < payment_count;
         ++payment) {
        const float coefficient = jamshidian_cashflow_coefficient(
            fixed_rate,
            schedule.accrual_fraction(payment),
            payment,
            payment_count
        );
        if (coefficient == 0.0f) continue;
        const OneFactorAffineBondCoefficients bond =
            provider.affine_bond_coefficients(
                parameters,
                exercise_time,
                schedule.payment_time(payment)
            );
        const float term = coefficient * expf(
            fmaf(-bond.B, state, bond.log_A)
        );
        coupon_bond += term;
        derivative = fmaf(-bond.B, term, derivative);
    }
    return {coupon_bond - 1.0f, derivative};
}

__device__ __forceinline__ float checked_jamshidian_boundary(
    float candidate,
    float residual
) {
    return isfinite(candidate)
            && isfinite(residual)
            && fabsf(residual) <= kJamshidianResidualTolerance
        ? candidate
        : nanf("");
}

// Solve sum_i c_i P(T_e,T_i;x*) = 1 with safeguarded Newton iterations.
template<
    typename Provider,
    typename Parameters,
    typename ScheduleView,
    std::uint32_t MaximumIterations =
        kMaximumJamshidianNewtonIterations
>
__device__ __forceinline__ float jamshidian_state_boundary(
    const Provider& provider,
    const Parameters& parameters,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    if (!schedule.valid()
        || !isfinite(exercise_time)
        || !isfinite(fixed_rate)
        || fixed_rate < 0.0f) {
        return nanf("");
    }

    const std::uint32_t payment_count = schedule.payment_count();
    const float log_payment_count = logf(
        static_cast<float>(payment_count)
    );
    float lower_state = FLT_MAX;
    float upper_state = -FLT_MAX;
    for (std::uint32_t payment = 0U;
         payment < payment_count;
         ++payment) {
        const float accrual_fraction = schedule.accrual_fraction(payment);
        const float coefficient = jamshidian_cashflow_coefficient(
            fixed_rate,
            accrual_fraction,
            payment,
            payment_count
        );
        const float payment_time = schedule.payment_time(payment);
        if (!isfinite(coefficient)
            || coefficient < 0.0f
            || !isfinite(accrual_fraction)
            || !(accrual_fraction > 0.0f)
            || !isfinite(payment_time)
            || !(payment_time > exercise_time)) {
            return nanf("");
        }
        if (coefficient == 0.0f) continue;

        const OneFactorAffineBondCoefficients bond =
            provider.affine_bond_coefficients(
                parameters, exercise_time, payment_time
            );
        const float log_weight = logf(coefficient) + bond.log_A;
        if (!isfinite(bond.B)
            || !(bond.B > 0.0f)
            || !isfinite(log_weight)) {
            return nanf("");
        }
        lower_state = fminf(
            lower_state, log_weight / bond.B
        );
        upper_state = fmaxf(
            upper_state,
            (log_weight + log_payment_count) / bond.B
        );
    }
    if (!isfinite(lower_state)
        || !isfinite(upper_state)
        || lower_state > upper_state) {
        return nanf("");
    }

    float state = 0.5f * (lower_state + upper_state);
    if constexpr (MaximumIterations > 0U) {
      for (std::uint32_t iteration = 0U;
           iteration < MaximumIterations;
           ++iteration) {
        const JamshidianBoundaryEvaluation evaluation =
            evaluate_jamshidian_boundary(
                provider,
                parameters,
                exercise_time,
                fixed_rate,
                schedule,
                state
            );
        const float residual = evaluation.residual;
        if (fabsf(residual) <= kJamshidianResidualTolerance) return state;
        if (residual > 0.0f)
            lower_state = state;
        else
            upper_state = state;

        const float midpoint = 0.5f * (lower_state + upper_state);
        float next_state = midpoint;
        if (isfinite(evaluation.derivative)
            && evaluation.derivative < 0.0f) {
            const float newton_state =
                state - residual / evaluation.derivative;
            const float quarter_width =
                0.25f * (upper_state - lower_state);
            if (isfinite(newton_state)
                && newton_state > lower_state + quarter_width
                && newton_state < upper_state - quarter_width) {
                next_state = newton_state;
            }
        }
        if (next_state == state || midpoint == lower_state
            || midpoint == upper_state) {
            const JamshidianBoundaryEvaluation final_evaluation =
                evaluate_jamshidian_boundary(
                    provider,
                    parameters,
                    exercise_time,
                    fixed_rate,
                    schedule,
                    midpoint
                );
            return checked_jamshidian_boundary(
                midpoint, final_evaluation.residual
            );
        }
        state = next_state;
      }
    }
    const float midpoint = 0.5f * (lower_state + upper_state);
    const JamshidianBoundaryEvaluation final_evaluation =
        evaluate_jamshidian_boundary(
            provider,
            parameters,
            exercise_time,
            fixed_rate,
            schedule,
            midpoint
        );
    return checked_jamshidian_boundary(
        midpoint, final_evaluation.residual
    );
}

// Evaluate P(T_e,T_i;x*) once the common state boundary is known.
template<typename Provider, typename Parameters>
__device__ __forceinline__ float jamshidian_bond_strike(
    const Provider& provider,
    const Parameters& parameters,
    float exercise_time,
    float payment_time,
    float state_boundary
) {
    return provider.zero_coupon_bond(
        parameters,
        state_boundary,
        exercise_time,
        payment_time
    );
}

// Decompose payer into bond puts and receiver into bond calls.
template<
    SwaptionSide Side,
    typename Provider,
    typename Parameters,
    typename State,
    typename ScheduleView
>
__device__ __forceinline__ float european_swaption_price(
    const Provider& provider,
    const Parameters& parameters,
    const State& state,
    float valuation_time,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    const float boundary = jamshidian_state_boundary(
        provider, parameters, exercise_time, fixed_rate, schedule
    );
    if (!isfinite(boundary)) return boundary;

    const auto option_context = provider.prepare_bond_option_context(
        parameters, state, valuation_time, exercise_time
    );
    constexpr float option_sign =
        Side == SwaptionSide::payer ? -1.0f : 1.0f;
    const std::uint32_t payment_count = schedule.payment_count();
    float price = 0.0f;
    for (std::uint32_t payment = 0U;
         payment < payment_count;
         ++payment) {
        const float coefficient = jamshidian_cashflow_coefficient(
            fixed_rate,
            schedule.accrual_fraction(payment),
            payment,
            payment_count
        );
        if (coefficient == 0.0f) continue;
        const float payment_time = schedule.payment_time(payment);
        const float bond_strike = jamshidian_bond_strike(
            provider,
            parameters,
            exercise_time,
            payment_time,
            boundary
        );
        price = fmaf(
            coefficient,
            provider.bond_option_price(
                option_context,
                parameters,
                state,
                option_sign,
                valuation_time,
                exercise_time,
                payment_time,
                bond_strike
            ),
            price
        );
    }
    return price;
}

}  // namespace ai_factory::workbench::fixed_income
