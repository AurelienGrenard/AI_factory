#include "ai_factory/cpu/hull_white/caplets.hpp"
#include "ai_factory/common/fixed_income/hull_white.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace ai_factory::cpu::hull_white {
void price_caplet_batch(const cuda::HullWhiteCapletRow* rows,std::size_t count,std::size_t paths,cuda::MonteCarloOutput* outputs){
    if(paths<2U)throw std::invalid_argument("Caplet pricing requires at least two paths.");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i){const auto&r=rows[i];const auto&p=r.product;const auto normals=simulation::philox_standard_normals(r.seed,2U*paths,0U);const double state_var=fixed_income::hull_white_state_variance(r.mean_reversion,r.volatility,p.fixing_time);const double integral_var=fixed_income::hull_white_integral_variance(r.mean_reversion,r.volatility,p.fixing_time);const double covariance=fixed_income::hull_white_state_integral_covariance(r.mean_reversion,r.volatility,p.fixing_time);const double state_scale=std::sqrt(state_var);const double integral_loading=covariance/state_scale;const double residual_scale=std::sqrt(std::max(integral_var-integral_loading*integral_loading,0.0));const double deterministic_integral=fixed_income::hull_white_deterministic_integral(p.fixing_time,r.mean_reversion,r.volatility,r.beta0,r.beta1,r.beta2,r.tau);const double maturity=p.fixing_time+p.accrual_period;const double bond_a=fixed_income::hull_white_bond_a(p.fixing_time,maturity,r.mean_reversion,r.volatility,r.beta0,r.beta1,r.beta2,r.tau);const double bond_b=fixed_income::hull_white_b(r.mean_reversion,p.accrual_period);double sum=0.0,sumsq=0.0;for(std::size_t path=0;path<paths;++path){const double z0=normals[2U*path],z1=normals[2U*path+1U];const double state=state_scale*z0;const double integral=integral_loading*z0+residual_scale*z1;const double bond=bond_a*std::exp(-bond_b*state);const double payoff=std::exp(-deterministic_integral-integral)*p.notional*std::max(1.0-(1.0+p.accrual_period*p.strike)*bond,0.0);sum+=payoff;sumsq+=payoff*payoff;}const double n=static_cast<double>(paths),mean=sum/n,var=(sumsq-n*mean*mean)/(n-1.0);outputs[i]={mean,std::sqrt(std::max(var,0.0)/n)};}
}
}
