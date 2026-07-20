#include "ai_factory/cpu/hull_white/zero_coupon_bonds.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"
namespace ai_factory::cpu::hull_white {
void price_zero_coupon_bond_batch(const cuda::HullWhiteZeroCouponBondRow* rows,std::size_t count,cuda::MonteCarloOutput* outputs){
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096U)
#endif
    for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i){const auto&row=rows[i];outputs[i]={row.product.notional*fixed_income::nelson_siegel_discount(row.product.maturity,row.beta0,row.beta1,row.beta2,row.tau),0.0};}
}
}
