#include "ai_factory/cpu/cir/interest_rate_swaps.hpp"
#include "ai_factory/common/fixed_income/cir.hpp"
#include "ai_factory/common/fixed_income/swaps.hpp"
namespace ai_factory::cpu::cir {
void price_interest_rate_swap_batch(const cuda::CirSwapRow* rows, std::size_t count, cuda::MonteCarloOutput* outputs) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096U)
#endif
    for (std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i) {
        const auto& row=rows[i];
        const auto discount=[&](double t){ return fixed_income::cir_bond_price(row.model.initial_rate,row.model.kappa,row.model.theta,row.model.volatility,t); };
        outputs[i]={fixed_income::swap_value(row.product,discount),0.0};
    }
}
}
