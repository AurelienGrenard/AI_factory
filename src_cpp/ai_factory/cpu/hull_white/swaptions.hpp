#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::hull_white {

void price_swaption_batch(
    const cuda::HullWhiteSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    cuda::MonteCarloOutput* outputs
);

}
