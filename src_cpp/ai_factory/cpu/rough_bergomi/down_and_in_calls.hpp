#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::rough_bergomi {

void price_down_and_in_call_batch(
    const cuda::RoughBergomiBarrierRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::rough_bergomi
