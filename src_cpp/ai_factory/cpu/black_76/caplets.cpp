#include "ai_factory/cpu/black_76/caplets.hpp"

#include "ai_factory/common/fixed_income/black_76.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"

#include <cmath>

namespace ai_factory::cpu::black_76 {

void price_caplet_batch(
    const cuda::Black76CapletRow* rows,
    std::size_t row_count,
    cuda::MonteCarloOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(row_count);
         ++index) {
        const auto& row = rows[index];
        const double fixing = row.product.fixing_time;
        const double payment = fixing + row.product.accrual_period;
        const double discount_fixing = fixed_income::nelson_siegel_discount(
            fixing, row.curve.beta0, row.curve.beta1,
            row.curve.beta2, row.curve.tau
        );
        const double discount_payment = fixed_income::nelson_siegel_discount(
            payment, row.curve.beta0, row.curve.beta1,
            row.curve.beta2, row.curve.tau
        );
        const double forward = (
            discount_fixing / discount_payment - 1.0
        ) / row.product.accrual_period;
        const double option = fixed_income::shifted_black_option(
            forward, row.product.strike, row.model.displacement,
            row.model.volatility * std::sqrt(fixing), 1
        );
        outputs[index] = {
            row.product.notional * row.product.accrual_period
                * discount_payment * option,
            0.0,
        };
    }
}

}  // namespace ai_factory::cpu::black_76
