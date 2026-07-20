#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::cir {

void price_bermudan_swaption_batch(
    const cuda::CirBermudanSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    double target_dt,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::cir
