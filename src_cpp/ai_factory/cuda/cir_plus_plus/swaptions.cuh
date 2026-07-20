#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cuda {
void price_cir_plus_plus_swaption_cuda(
    const CirPlusPlusSwaptionRow*, std::size_t, std::size_t, double,
    MonteCarloOutput*, CudaTiming*
);
}
