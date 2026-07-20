#include "ai_factory/cuda/rough_bergomi/up_and_out_calls.cuh"
#include "ai_factory/cuda/rough_bergomi/barrier_calls.cuh"
namespace ai_factory::cuda { namespace { struct Tag {}; }
void price_rough_bergomi_up_and_out_call_cuda(const RoughBergomiBarrierRow* r,std::size_t n,std::size_t p,std::size_t s,MonteCarloOutput* o,CudaTiming* t){rough_bergomi_barrier_detail::run<true,false,Tag>(r,n,p,s,o,t);}}
