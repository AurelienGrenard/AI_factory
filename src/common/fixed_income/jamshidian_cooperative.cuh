// Cooperative one-factor Jamshidian decomposition over fixed-leg cashflows.
#pragma once

#include "common/fixed_income/analytics_concepts.cuh"
#include "common/fixed_income/jamshidian.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::fixed_income {

inline std::size_t cooperative_jamshidian_shared_memory_bytes(
    std::uint32_t payment_capacity
) {
    constexpr std::size_t arrays_per_payment = 3U;
    const std::size_t capacity = payment_capacity;
    if (capacity > std::numeric_limits<std::size_t>::max()
            / (arrays_per_payment * sizeof(float))) {
        throw std::overflow_error(
            "The cooperative Jamshidian workspace exceeds size_t."
        );
    }
    return arrays_per_payment * capacity * sizeof(float);
}

struct CooperativeJamshidianWorkspace {
    float* log_A;
    float* B;
    float* option_values;
};

template<typename ScheduleView>
__device__ __forceinline__ JamshidianBoundaryEvaluation
evaluate_jamshidian_boundary_from_coefficients(
    float fixed_rate,
    const ScheduleView& schedule,
    const CooperativeJamshidianWorkspace& workspace,
    float state
) {
    const std::uint32_t payment_count = schedule.payment_count();
    float coupon_bond = 0.0f;
    float derivative = 0.0f;
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
        const float term = coefficient * expf(
            fmaf(
                -workspace.B[payment],
                state,
                workspace.log_A[payment]
            )
        );
        coupon_bond += term;
        derivative = fmaf(
            -workspace.B[payment], term, derivative
        );
    }
    return {coupon_bond - 1.0f, derivative};
}

__device__ __forceinline__ CooperativeJamshidianWorkspace
make_cooperative_jamshidian_workspace(
    std::byte* storage,
    std::uint32_t payment_capacity
) {
    float* values = reinterpret_cast<float*>(storage);
    return {
        values,
        values + payment_capacity,
        values + 2U * payment_capacity,
    };
}

template<
    typename ScheduleView,
    std::uint32_t MaximumIterations =
        kMaximumJamshidianNewtonIterations
>
__device__ __forceinline__ float
jamshidian_state_boundary_from_coefficients(
    float fixed_rate,
    const ScheduleView& schedule,
    const CooperativeJamshidianWorkspace& workspace
) {
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
        const float log_A = workspace.log_A[payment];
        const float B = workspace.B[payment];
        if (!isfinite(coefficient)
            || coefficient < 0.0f
            || !isfinite(accrual_fraction)
            || !(accrual_fraction > 0.0f)
            || !isfinite(log_A)
            || !isfinite(B)
            || !(B > 0.0f)) {
            return nanf("");
        }
        if (coefficient == 0.0f) continue;
        const float log_weight = logf(coefficient) + log_A;
        lower_state = fminf(lower_state, log_weight / B);
        upper_state = fmaxf(
            upper_state,
            (log_weight + log_payment_count) / B
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
            evaluate_jamshidian_boundary_from_coefficients(
                fixed_rate, schedule, workspace, state
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
                evaluate_jamshidian_boundary_from_coefficients(
                    fixed_rate, schedule, workspace, midpoint
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
        evaluate_jamshidian_boundary_from_coefficients(
            fixed_rate, schedule, workspace, midpoint
        );
    return checked_jamshidian_boundary(
        midpoint, final_evaluation.residual
    );
}

template<
    SwaptionSide Side,
    typename Provider,
    typename Parameters,
    typename State,
    typename ScheduleView
>
__device__ __forceinline__ float cooperative_european_swaption_price(
    const Provider& provider,
    const Parameters& parameters,
    const State& state,
    float valuation_time_years,
    float exercise_time,
    float fixed_rate,
    const ScheduleView& schedule,
    std::byte* workspace_storage,
    std::uint32_t workspace_capacity
) {
    static_assert(
        JamshidianAnalyticsProvider<Provider, Parameters, State>
    );
    using BondOptionContext = decltype(
        provider.prepare_bond_option_context(
            parameters,
            state,
            valuation_time_years,
            exercise_time
        )
    );
    static_assert(std::is_trivially_copyable_v<BondOptionContext>);

    const std::uint32_t payment_count = schedule.payment_count();
    if (!schedule.valid()
        || payment_count > workspace_capacity
        || workspace_storage == nullptr
        || !isfinite(exercise_time)
        || !isfinite(fixed_rate)
        || fixed_rate < 0.0f) {
        return threadIdx.x == 0U ? nanf("") : 0.0f;
    }
    const CooperativeJamshidianWorkspace workspace =
        make_cooperative_jamshidian_workspace(
            workspace_storage,
            workspace_capacity
        );
    for (std::uint32_t payment = threadIdx.x;
         payment < payment_count;
         payment += blockDim.x) {
        const float payment_time = schedule.payment_time(payment);
        if (!isfinite(payment_time) || !(payment_time > exercise_time)) {
            workspace.log_A[payment] = nanf("");
            workspace.B[payment] = nanf("");
            continue;
        }
        const OneFactorAffineBondCoefficients bond =
            provider.affine_bond_coefficients(
                parameters,
                exercise_time,
                payment_time
            );
        workspace.log_A[payment] = bond.log_A;
        workspace.B[payment] = bond.B;
    }
    __syncthreads();

    __shared__ float state_boundary;
    __shared__ BondOptionContext option_context;
    if (threadIdx.x == 0U) {
        state_boundary = payment_count == 1U
            ? 0.0f
            : jamshidian_state_boundary_from_coefficients(
                  fixed_rate,
                  schedule,
                  workspace
              );
        if (isfinite(state_boundary)) {
            option_context = provider.prepare_bond_option_context(
                parameters,
                state,
                valuation_time_years,
                exercise_time
            );
        }
    }
    __syncthreads();
    if (!isfinite(state_boundary)) {
        return threadIdx.x == 0U ? state_boundary : 0.0f;
    }

    constexpr float option_sign =
        Side == SwaptionSide::payer ? -1.0f : 1.0f;
    for (std::uint32_t payment = threadIdx.x;
         payment < payment_count;
         payment += blockDim.x) {
        const float coefficient = jamshidian_cashflow_coefficient(
            fixed_rate,
            schedule.accrual_fraction(payment),
            payment,
            payment_count
        );
        if (coefficient == 0.0f) {
            workspace.option_values[payment] = 0.0f;
            continue;
        }
        const float payment_time = schedule.payment_time(payment);
        const float bond_strike = payment_count == 1U
            ? 1.0f / coefficient
            : provider.zero_coupon_bond(
                  parameters,
                  state_boundary,
                  exercise_time,
                  payment_time
              );
        workspace.option_values[payment] = provider.bond_option_price(
            option_context,
            parameters,
            state,
            option_sign,
            valuation_time_years,
            exercise_time,
            payment_time,
            bond_strike
        );
    }
    __syncthreads();

    if (threadIdx.x != 0U) return 0.0f;
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
        price = fmaf(
            coefficient,
            workspace.option_values[payment],
            price
        );
    }
    return price;
}

}  // namespace ai_factory::workbench::fixed_income
