#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::cir_plus_plus {

void price_bermudan_swaption_batch(
    const cuda::CirPlusPlusBermudanSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    double target_dt,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::cir_plus_plus
