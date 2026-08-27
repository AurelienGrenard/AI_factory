// Discounted single-curve forwards, swap rates, and swap values.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::fixed_income {

// Adapt already-converted analytical schedule arrays to the common interface.
struct FixedLegScheduleView {
    const float* payment_times;
    const float* accrual_fractions;
    std::uint32_t count;

    __device__ __forceinline__ float payment_time(
        std::uint32_t payment
    ) const {
        return payment_times[payment];
    }

    __device__ __forceinline__ float accrual_fraction(
        std::uint32_t payment
    ) const {
        return accrual_fractions[payment];
    }

    __device__ __forceinline__ std::uint32_t payment_count() const {
        return count;
    }
};

// Adapt business-day payment arrays without depending on a product type.
struct BusinessDayFixedLegScheduleView {
    const std::uint32_t* payment_times_days;
    const float* accrual_fractions;
    std::uint32_t count;
    float time_day_fraction;
    std::size_t stride = 1U;

    __device__ __forceinline__ bool valid() const {
        return count > 0U
            && payment_times_days != nullptr
            && accrual_fractions != nullptr
            && isfinite(time_day_fraction)
            && time_day_fraction > 0.0f;
    }

    __device__ __forceinline__ float payment_time(
        std::uint32_t payment
    ) const {
        return static_cast<float>(payment_times_days[payment * stride])
            * time_day_fraction;
    }

    __device__ __forceinline__ float accrual_fraction(
        std::uint32_t payment
    ) const {
        return accrual_fractions[payment * stride];
    }

    __device__ __forceinline__ std::uint32_t payment_count() const {
        return count;
    }
};

static_assert(sizeof(BusinessDayFixedLegScheduleView) == 32U);

// Build L(t;T_0,T_1) from two zero-coupon bonds.
template<typename BondProvider, typename Parameters, typename State>
__device__ __forceinline__ float forward_rate(
    const BondProvider& bonds,
    const Parameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    float end_time,
    float accrual_fraction
) {
    const float start_bond = bonds.zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    const float end_bond = bonds.zero_coupon_bond(
        parameters, state, valuation_time, end_time
    );
    return (start_bond / end_bond - 1.0f) / accrual_fraction;
}

// Return the fixed-leg annuity and final zero-coupon in one schedule pass.
template<
    typename BondProvider,
    typename Parameters,
    typename State,
    typename ScheduleView
>
__device__ __forceinline__ void fixed_leg_terms(
    const BondProvider& bonds,
    const Parameters& parameters,
    const State& state,
    float valuation_time,
    const ScheduleView& schedule,
    float& annuity,
    float& final_bond
) {
    annuity = 0.0f;
    final_bond = 0.0f;
    const std::uint32_t payment_count = schedule.payment_count();
    for (std::uint32_t payment = 0U;
         payment < payment_count;
         ++payment) {
        const float current_bond = bonds.zero_coupon_bond(
            parameters,
            state,
            valuation_time,
            schedule.payment_time(payment)
        );
        annuity = fmaf(
            schedule.accrual_fraction(payment), current_bond, annuity
        );
        final_bond = current_bond;
    }
}

// Return S(t) = (P(t,T_0)-P(t,T_N))/A(t) in the single-curve framework.
template<
    typename BondProvider,
    typename Parameters,
    typename State,
    typename ScheduleView
>
__device__ __forceinline__ float swap_rate(
    const BondProvider& bonds,
    const Parameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    const ScheduleView& schedule
) {
    float annuity = 0.0f;
    float final_bond = 0.0f;
    fixed_leg_terms(
        bonds,
        parameters,
        state,
        valuation_time,
        schedule,
        annuity,
        final_bond
    );
    const float start_bond = bonds.zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    return (start_bond - final_bond) / annuity;
}

// Return V_payer(t)/N = P(t,T_0)-P(t,T_N)-K A(t).
template<
    typename BondProvider,
    typename Parameters,
    typename State,
    typename ScheduleView
>
__device__ __forceinline__ float payer_swap_value(
    const BondProvider& bonds,
    const Parameters& parameters,
    const State& state,
    float valuation_time,
    float start_time,
    float fixed_rate,
    const ScheduleView& schedule
) {
    float annuity = 0.0f;
    float final_bond = 0.0f;
    fixed_leg_terms(
        bonds,
        parameters,
        state,
        valuation_time,
        schedule,
        annuity,
        final_bond
    );
    const float start_bond = bonds.zero_coupon_bond(
        parameters, state, valuation_time, start_time
    );
    return start_bond - final_bond - fixed_rate * annuity;
}

}  // namespace ai_factory::workbench::fixed_income
