#include "ai_factory/cpu/cir/zero_coupon_bonds.hpp"
#include "ai_factory/common/fixed_income/cir.hpp"
namespace ai_factory::cpu::cir {
void price_zero_coupon_bond_batch(const cuda::CirZeroCouponBondRow* rows,std::size_t count,cuda::MonteCarloOutput* outputs){
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count >= 4096U)
#endif
    for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i){const auto&row=rows[i];const auto&m=row.model;outputs[i]={row.product.notional*fixed_income::cir_bond_price(m.initial_rate,m.kappa,m.theta,m.volatility,row.product.maturity),0.0};}
}
}
