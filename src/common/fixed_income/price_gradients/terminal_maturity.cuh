// Gradient-only views that move the last contractual fixed-income date.
#pragma once

#include "product/european_swaption/pricing_row.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::fixed_income::price_gradients {

template<typename Schedule>
struct TerminalPaymentScheduleView {
    Schedule central;
    float terminal_payment_time;
    float terminal_prefix_time;
    bool override_terminal;

    __device__ __forceinline__ bool valid() const {
        return central.valid()
            && central.payment_count() > 0U
            && ::isfinite(terminal_payment_time)
            && ::isfinite(terminal_prefix_time)
            && terminal_payment_time > terminal_prefix_time;
    }

    __device__ __forceinline__ float payment_time(
        std::uint32_t payment
    ) const {
        return override_terminal
                && payment + 1U == central.payment_count()
            ? terminal_payment_time
            : central.payment_time(payment);
    }

    __device__ __forceinline__ float accrual_fraction(
        std::uint32_t payment
    ) const {
        return override_terminal
                && payment + 1U == central.payment_count()
            ? terminal_payment_time - terminal_prefix_time
            : central.accrual_fraction(payment);
    }

    __device__ __forceinline__ std::uint32_t payment_count() const {
        return central.payment_count();
    }
};

template<typename PreparedRow>
struct TerminalMaturityRow {
    using Type = PreparedRow;
};

template<typename Model, typename Schedule>
struct TerminalMaturityRow<
    fixed_income::PreparedEuropeanSwaptionRow<Model, Schedule>
> {
    using Type = fixed_income::PreparedEuropeanSwaptionRow<
        Model,
        TerminalPaymentScheduleView<Schedule>
    >;
};

template<typename PreparedRow>
using TerminalMaturityPreparedRow =
    typename TerminalMaturityRow<PreparedRow>::Type;

template<typename PreparedRow>
__device__ __forceinline__ TerminalMaturityPreparedRow<PreparedRow>
apply_terminal_maturity(
    PreparedRow row,
    float maturity_years,
    bool override_terminal
) {
    if constexpr (requires { row.payment_time_years; }) {
        if (override_terminal) row.payment_time_years = maturity_years;
        return row;
    } else if constexpr (requires { row.bond_maturity_years; }) {
        if (override_terminal) row.bond_maturity_years = maturity_years;
        return row;
    } else {
        const auto count = row.schedule.payment_count();
        const float prefix = count > 1U
            ? row.schedule.payment_time(count - 2U)
            : row.exercise_time_years;
        return {
            row.model,
            row.notional,
            row.strike,
            row.exercise_time_years,
            {row.schedule, maturity_years, prefix, override_terminal},
        };
    }
}

}  // namespace ai_factory::workbench::fixed_income::price_gradients
