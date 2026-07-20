#include "ai_factory/cpu/g2_plus_plus/interest_rate_swaps.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"
#include "ai_factory/common/fixed_income/swaps.hpp"
namespace ai_factory::cpu::g2_plus_plus {
void price_interest_rate_swap_batch(const cuda::G2PlusPlusSwapRow* rows,std::size_t count,cuda::MonteCarloOutput* outputs){
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096U)
#endif
for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i){const auto&r=rows[i];const auto discount=[&](double t){return fixed_income::nelson_siegel_discount(t,r.curve.beta0,r.curve.beta1,r.curve.beta2,r.curve.tau);};outputs[i]={fixed_income::swap_value(r.product,discount),0.0};}}
}
