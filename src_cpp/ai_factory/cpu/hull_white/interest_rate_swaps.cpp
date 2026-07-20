#include "ai_factory/cpu/hull_white/interest_rate_swaps.hpp"

#include "ai_factory/common/fixed_income/nelson_siegel.hpp"
#include "ai_factory/common/fixed_income/swaps.hpp"

namespace ai_factory::cpu::hull_white {

void price_interest_rate_swap_batch(
    const cuda::HullWhiteSwapRow* rows,
    std::size_t row_count,
    cuda::MonteCarloOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(row_count >= 4096U)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(row_count); ++index) {
        const auto& row = rows[index];
        const auto discount = [&](double maturity) {
            return fixed_income::nelson_siegel_discount(
                maturity, row.beta0, row.beta1, row.beta2, row.tau
            );
        };
        outputs[index] = {fixed_income::swap_value(row.product, discount), 0.0};
    }
}

}
