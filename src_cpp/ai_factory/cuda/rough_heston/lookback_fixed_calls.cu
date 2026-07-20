#include "ai_factory/cuda/rough_heston/lookback_fixed_calls.cuh"
#include "ai_factory/cuda/rough_heston/pathwise_pricing.cuh"
namespace ai_factory::cuda { namespace { struct PriceTag {}; struct DeltaTag {}; }
void price_rough_heston_lookback_fixed_cuda(const RoughHestonRow* r,std::size_t n,std::size_t p,std::size_t s,MonteCarloOutput* o,CudaTiming* t){rough_heston_pathwise::run<PriceTag,rough_heston_pathwise::Statistic::Maximum,false,false>(r,n,p,s,0.0,o,t);}
void price_rough_heston_lookback_fixed_delta_crn_cuda(const RoughHestonRow* r,std::size_t n,std::size_t p,std::size_t s,double b,PriceDeltaOutput* o,CudaTiming* t){rough_heston_pathwise::run<DeltaTag,rough_heston_pathwise::Statistic::Maximum,false,true>(r,n,p,s,b,o,t);}}
