// Gradient-only views that move the last contractual fixed-income date.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

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

    __device__ __forceinline__ static Type apply(
        PreparedRow row,
        float maturity_years,
        bool override_terminal
    ) {
        if constexpr (requires { row.payment_time_years; }) {
            if (override_terminal) row.payment_time_years = maturity_years;
        } else if constexpr (requires { row.bond_maturity_years; }) {
            if (override_terminal) row.bond_maturity_years = maturity_years;
        } else {
            static_assert(
                sizeof(PreparedRow) == 0,
                "A terminal-maturity row adapter is required for this product."
            );
        }
        return row;
    }
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
    return TerminalMaturityRow<PreparedRow>::apply(
        row, maturity_years, override_terminal
    );
}

}  // namespace ai_factory::workbench::fixed_income::price_gradients
