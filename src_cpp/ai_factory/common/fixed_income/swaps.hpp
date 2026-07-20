#pragma once

#include "ai_factory/cuda/common/types.cuh"

#ifdef __CUDACC__
#define AI_FACTORY_HD __host__ __device__ __forceinline__
#else
#define AI_FACTORY_HD inline
#endif

namespace ai_factory::fixed_income {

template <typename DiscountFunction>
AI_FACTORY_HD double swap_value(
    const cuda::SwapTerms& terms,
    DiscountFunction discount
) {
    double annuity = 0.0;
    for (int payment = 1; payment <= terms.payment_count; ++payment) {
        annuity += terms.accrual_period
                   * discount(terms.start_time + payment * terms.accrual_period);
    }
    const double maturity = terms.start_time
                            + terms.payment_count * terms.accrual_period;
    const double payer = discount(terms.start_time) - discount(maturity)
                         - terms.fixed_rate * annuity;
    return terms.notional * static_cast<double>(terms.direction) * payer;
}

}  // namespace ai_factory::fixed_income

#undef AI_FACTORY_HD
