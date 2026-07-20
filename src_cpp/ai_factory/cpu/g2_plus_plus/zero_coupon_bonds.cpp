#include "ai_factory/cpu/g2_plus_plus/zero_coupon_bonds.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"
namespace ai_factory::cpu::g2_plus_plus {
void price_zero_coupon_bond_batch(const cuda::G2PlusPlusZeroCouponBondRow* rows,std::size_t count,cuda::MonteCarloOutput* outputs){
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096U)
#endif
for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i){const auto&r=rows[i];outputs[i]={r.product.notional*fixed_income::nelson_siegel_discount(r.product.maturity,r.curve.beta0,r.curve.beta1,r.curve.beta2,r.curve.tau),0.0};}}
}
