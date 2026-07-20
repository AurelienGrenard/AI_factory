#include "ai_factory/cpu/cir_plus_plus/caplets.hpp"
#include "ai_factory/common/fixed_income/cir_plus_plus.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace ai_factory::cpu::cir_plus_plus {namespace {constexpr double kPsi=1.5;}
void price_caplet_batch(const cuda::CirPlusPlusCapletRow*rows,std::size_t count,std::size_t paths,double target_dt,cuda::MonteCarloOutput*outputs){if(paths<2U)throw std::invalid_argument("Caplet pricing requires at least two paths.");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
for(std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i){const auto&r=rows[i];const auto&m=r.model;const auto&p=r.product;const std::size_t steps=std::max<std::size_t>(1U,static_cast<std::size_t>(std::llround(p.fixing_time/target_dt)));const double dt=p.fixing_time/static_cast<double>(steps),decay=std::exp(-m.kappa*dt),omd=1.0-decay,vol2=m.volatility*m.volatility,maturity=p.fixing_time+p.accrual_period;const double ba=fixed_income::cir_plus_plus_bond_a(p.fixing_time,maturity,m.initial_factor,m.kappa,m.theta,m.volatility,r.curve.beta0,r.curve.beta1,r.curve.beta2,r.curve.tau),bb=fixed_income::cir_bond_b(m.kappa,m.volatility,p.accrual_period);double sum=0,sumsq=0;for(std::size_t path=0;path<paths;++path){double factor=m.initial_factor,integral=0;simulation::PhiloxNormalSequence normals(r.seed,path);simulation::PhiloxUniformSequence uniforms(r.seed,path+paths);for(std::size_t step=0;step<steps;++step){const double previous=factor,mean=m.theta+(factor-m.theta)*decay,var=factor*vol2*decay*omd/m.kappa+m.theta*vol2*omd*omd/(2*m.kappa),psi=var/(mean*mean),z=normals.next(),u=uniforms.next();if(psi<=kPsi){const double inv=1/psi,b2=2*inv-1+std::sqrt(2*inv)*std::sqrt(std::max(2*inv-1,0.0)),scale=mean/(1+b2),shift=std::sqrt(b2)+z;factor=scale*shift*shift;}else{const double probability=(psi-1)/(psi+1),beta=(1-probability)/mean;factor=u<=probability?0.0:std::log((1-probability)/(1-u))/beta;}integral+=0.5*(previous+factor)*dt;}const double bond=ba*std::exp(-bb*factor),discount=fixed_income::cir_plus_plus_path_discount(integral,p.fixing_time,m.initial_factor,m.kappa,m.theta,m.volatility,r.curve.beta0,r.curve.beta1,r.curve.beta2,r.curve.tau),payoff=discount*p.notional*std::max(1-(1+p.accrual_period*p.strike)*bond,0.0);sum+=payoff;sumsq+=payoff*payoff;}const double n=static_cast<double>(paths),mean=sum/n,var=(sumsq-n*mean*mean)/(n-1);outputs[i]={mean,std::sqrt(std::max(var,0.0)/n)};}}
}
