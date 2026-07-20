#pragma once

#include "ai_factory/cuda/rough_heston/api.cuh"

namespace ai_factory::cuda {

void price_rough_heston_american_put_cuda(
    const RoughHestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

}  // namespace ai_factory::cuda
