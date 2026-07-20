#include "ai_factory/cuda/rough_heston/volatility_swaps.cuh"
#include "ai_factory/cuda/rough_heston/pathwise_pricing.cuh"
namespace ai_factory::cuda { namespace { struct PriceTag {}; }
void price_rough_heston_volatility_swap_cuda(const RoughHestonRow* r,std::size_t n,std::size_t p,std::size_t s,MonteCarloOutput* o,CudaTiming* t){rough_heston_pathwise::run<PriceTag,rough_heston_pathwise::Statistic::RealizedVolatility,true,false>(r,n,p,s,0.0,o,t);}}
