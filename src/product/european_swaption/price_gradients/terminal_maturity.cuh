// Adapt the common terminal-payment view to a European-swaption row.
#pragma once

#include "common/fixed_income/price_gradients/terminal_maturity.cuh"
#include "product/european_swaption/pricing_row.cuh"

namespace ai_factory::workbench::fixed_income::price_gradients {

template<typename Model, typename Schedule>
struct TerminalMaturityRow<
    fixed_income::PreparedEuropeanSwaptionRow<Model, Schedule>
> {
    using Type = fixed_income::PreparedEuropeanSwaptionRow<
        Model, TerminalPaymentScheduleView<Schedule>
    >;

    __device__ __forceinline__ static Type apply(
        fixed_income::PreparedEuropeanSwaptionRow<Model, Schedule> row,
        float maturity_years,
        bool override_terminal
    ) {
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
};

}  // namespace ai_factory::workbench::fixed_income::price_gradients
