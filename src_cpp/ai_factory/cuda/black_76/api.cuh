#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_black_76_caplet_cuda(
    const Black76CapletRow*, std::size_t, MonteCarloOutput*, CudaTiming*
);
void price_black_76_swaption_cuda(
    const Black76SwaptionRow*, std::size_t, MonteCarloOutput*, CudaTiming*
);

}  // namespace ai_factory::cuda
