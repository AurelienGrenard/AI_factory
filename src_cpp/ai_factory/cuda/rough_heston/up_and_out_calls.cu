#include "ai_factory/cuda/rough_heston/up_and_out_calls.cuh"
#include "ai_factory/cuda/common/barrier_pricing.cuh"
#include "ai_factory/cuda/rough_heston/api.cuh"
#include "ai_factory/cuda/rough_heston/dynamics.cuh"
namespace ai_factory::cuda { namespace { struct Tag {}; }
void price_rough_heston_up_and_out_call_cuda(const RoughHestonBarrierRow* r,std::size_t n,std::size_t p,std::size_t s,MonteCarloOutput* o,CudaTiming* t){barrier_detail::run<RoughHestonBarrierRow,rough_heston_detail::BarrierSimulator,false,Tag>(r,n,p,s,o,t);}}
