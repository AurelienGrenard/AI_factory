#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_rough_bergomi_down_and_out_call_cuda(
    const RoughBergomiBarrierRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* outputs,
    CudaTiming* timing
);

}  // namespace ai_factory::cuda
