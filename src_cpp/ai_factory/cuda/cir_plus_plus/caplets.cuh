#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cuda {
void price_cir_plus_plus_caplet_cuda(const CirPlusPlusCapletRow*,std::size_t,std::size_t,double,MonteCarloOutput*,CudaTiming*);
}
