#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_hull_white_bermudan_swaption_cuda(
    const HullWhiteBermudanSwaptionRow*, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);

}  // namespace ai_factory::cuda
