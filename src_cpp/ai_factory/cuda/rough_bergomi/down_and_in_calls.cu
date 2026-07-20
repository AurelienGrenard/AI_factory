#include "ai_factory/cuda/rough_bergomi/down_and_in_calls.cuh"
#include "ai_factory/cuda/rough_bergomi/barrier_calls.cuh"
namespace ai_factory::cuda { namespace { struct Tag {}; }
void price_rough_bergomi_down_and_in_call_cuda(const RoughBergomiBarrierRow* r,std::size_t n,std::size_t p,std::size_t s,MonteCarloOutput* o,CudaTiming* t){rough_bergomi_barrier_detail::run<false,true,Tag>(r,n,p,s,o,t);}}
