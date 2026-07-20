#include "ai_factory/cuda/black_scholes/down_and_out_calls.cuh"
#include "ai_factory/cuda/black_scholes/barrier_calls.cuh"
namespace ai_factory::cuda { namespace { struct Tag {}; }
void price_black_scholes_down_and_out_call_cuda(const BlackScholesBarrierRow* r, std::size_t n, std::size_t p, std::size_t s, MonteCarloOutput* o, CudaTiming* t) { barrier_detail::run<BlackScholesBarrierRow, black_scholes_detail::BarrierSimulator<false>, false, Tag>(r,n,p,s,o,t); }
}
