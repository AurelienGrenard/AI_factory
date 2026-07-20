#include "ai_factory/cpu/black_76/swaptions.hpp"

#include "ai_factory/common/fixed_income/black_76.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"

#include <cmath>

namespace ai_factory::cpu::black_76 {

void price_swaption_batch(
    const cuda::Black76SwaptionRow* rows,
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
        const auto& product = row.product;
        const double discount_start = fixed_income::nelson_siegel_discount(
            product.expiry, row.curve.beta0, row.curve.beta1,
            row.curve.beta2, row.curve.tau
        );
        double annuity = 0.0;
        double discount_end = discount_start;
        for (int payment = 1; payment <= product.payment_count; ++payment) {
            discount_end = fixed_income::nelson_siegel_discount(
                product.expiry + payment * product.accrual_period,
                row.curve.beta0, row.curve.beta1,
                row.curve.beta2, row.curve.tau
            );
            annuity += product.accrual_period * discount_end;
        }
        const double forward_swap = (
            discount_start - discount_end
        ) / annuity;
        const double option = fixed_income::shifted_black_option(
            forward_swap, product.fixed_rate, row.model.displacement,
            row.model.volatility * std::sqrt(product.expiry),
            product.direction
        );
        outputs[index] = {product.notional * annuity * option, 0.0};
    }
}

}  // namespace ai_factory::cpu::black_76
