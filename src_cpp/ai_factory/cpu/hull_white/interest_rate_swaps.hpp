#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::hull_white {

void price_interest_rate_swap_batch(
    const cuda::HullWhiteSwapRow* rows,
    std::size_t row_count,
    cuda::MonteCarloOutput* outputs
);

}
