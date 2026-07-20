#include "ai_factory/cpu/g2_plus_plus/swaptions.hpp"
#include "ai_factory/common/fixed_income/g2_plus_plus.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace ai_factory::cpu::g2_plus_plus {namespace {constexpr int kMaxPayments=20;}
void price_swaption_batch(const cuda::G2PlusPlusSwaptionRow* rows,std::size_t count,std::size_t paths,cuda::MonteCarloOutput* outputs){if(paths<2)throw std::invalid_argument("G2++ swaption requires two paths.");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
for(std::ptrdiff_t ri=0;ri<static_cast<std::ptrdiff_t>(count);++ri){const auto&r=rows[ri];const auto&m=r.model;const auto&p=r.product;const auto transition=fixed_income::make_g2_transition(m.mean_reversion_x,m.volatility_x,m.mean_reversion_y,m.volatility_y,m.rho,p.expiry);std::array<double,kMaxPayments> ba{},bbx{},bby{};for(int j=0;j<p.payment_count;++j){const double maturity=p.expiry+(j+1)*p.accrual_period;ba[j]=fixed_income::g2_bond_price(0,0,p.expiry,maturity,m.mean_reversion_x,m.volatility_x,m.mean_reversion_y,m.volatility_y,m.rho,r.curve.beta0,r.curve.beta1,r.curve.beta2,r.curve.tau);bbx[j]=fixed_income::ou_b(m.mean_reversion_x,maturity-p.expiry);bby[j]=fixed_income::ou_b(m.mean_reversion_y,maturity-p.expiry);}double sum=0,sumsq=0;for(std::size_t path=0;path<paths;++path){double x=0,y=0,ix=0,iy=0;fixed_income::apply_g2_transition(transition,simulation::philox_standard_normal(r.seed,0U,4U*path),simulation::philox_standard_normal(r.seed,0U,4U*path+1U),simulation::philox_standard_normal(r.seed,0U,4U*path+2U),simulation::philox_standard_normal(r.seed,0U,4U*path+3U),x,y,ix,iy);double annuity=0,end=1;for(int j=0;j<p.payment_count;++j){const double bond=ba[j]*std::exp(-bbx[j]*x-bby[j]*y);annuity+=p.accrual_period*bond;end=bond;}const double swap=static_cast<double>(p.direction)*(1-end-p.fixed_rate*annuity);const double discount=fixed_income::g2_path_discount(ix,iy,p.expiry,m.mean_reversion_x,m.volatility_x,m.mean_reversion_y,m.volatility_y,m.rho,r.curve.beta0,r.curve.beta1,r.curve.beta2,r.curve.tau);const double payoff=discount*p.notional*std::max(swap,0.0);sum+=payoff;sumsq+=payoff*payoff;}const double n=static_cast<double>(paths),mean=sum/n,var=(sumsq-n*mean*mean)/(n-1);outputs[ri]={mean,std::sqrt(std::max(var,0.0)/n)};}}
}
