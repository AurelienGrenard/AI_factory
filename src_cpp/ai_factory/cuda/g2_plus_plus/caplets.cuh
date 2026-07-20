#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cuda {
void price_g2_plus_plus_caplet_cuda(const G2PlusPlusCapletRow*,std::size_t,std::size_t,MonteCarloOutput*,CudaTiming*);
}
