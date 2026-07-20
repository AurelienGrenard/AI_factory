#include "ai_factory/cuda/heston/down_and_in_calls.cuh"
#include "ai_factory/cuda/heston/barrier_calls.cuh"
namespace ai_factory::cuda { namespace { struct Tag {}; }
void price_heston_down_and_in_call_cuda(const HestonBarrierRow* r,std::size_t n,std::size_t p,std::size_t s,MonteCarloOutput* o,CudaTiming* t){barrier_detail::run<HestonBarrierRow,heston_detail::BarrierSimulator<false>,true,Tag>(r,n,p,s,o,t);}}
